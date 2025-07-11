#!/usr/bin/env python3
"""
MusiCoT Training Script
======================

This script implements the complete training pipeline for MusiCoT 
(Chain-of-Musical-Thought) music generation model.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from tqdm import tqdm
import wandb

# Local imports
from models.musicot_models import (
    CLAPQuantizer, MusiCoTSemanticLM, DualSamplingStrategy, 
    MusicStructureAnalyzer, MusiCoTLoss
)
from data.dataset import MusiCoTDataset, create_dataloaders
from utils.clap_model import CLAPModel
from utils.audio_processing import AudioProcessor
from utils.evaluation import MusiCoTEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MusiCoTTrainer:
    """Main trainer class for MusiCoT"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'])
        self.local_rank = config.get('local_rank', 0)
        
        # Initialize models
        self._init_models()
        
        # Initialize datasets
        self._init_datasets()
        
        # Initialize training components
        self._init_training_components()
        
        # Initialize logging
        self._init_logging()
        
    def _init_models(self):
        """Initialize all model components"""
        logger.info("Initializing models...")
        
        # CLAP model for embeddings
        self.clap_model = CLAPModel(
            embed_dim=self.config['clap']['embed_dim'],
            audio_length=self.config['clap']['audio_length'],
            sample_rate=self.config['clap']['sample_rate']
        )
        
        # RVQ model for CLAP quantization
        self.rvq_model = CLAPQuantizer(
            embed_dim=self.config['clap']['embed_dim'],
            num_quantizers=self.config['rvq']['num_quantizers'],
            codebook_size=self.config['rvq']['codebook_size'],
            commitment_weight=self.config['rvq']['commitment_weight']
        )
        
        # Semantic Language Model
        self.semantic_lm = MusiCoTSemanticLM(
            model_name=self.config['semantic_lm']['model_name'],
            vocab_size_extension=self.config['semantic_lm']['vocab_size_extension'],
            max_sequence_length=self.config['semantic_lm']['max_sequence_length']
        )
        
        # Move models to device
        self.clap_model = self.clap_model.to(self.device)
        self.rvq_model = self.rvq_model.to(self.device)
        self.semantic_lm = self.semantic_lm.to(self.device)
        
        # Setup distributed training if needed
        if self.config.get('distributed', False):
            self.semantic_lm = DDP(
                self.semantic_lm, 
                device_ids=[self.local_rank],
                find_unused_parameters=True
            )
            self.rvq_model = DDP(
                self.rvq_model,
                device_ids=[self.local_rank]
            )
        
        # Dual sampling strategy
        self.sampling_strategy = DualSamplingStrategy(
            cot_temperature=self.config['sampling']['cot_temperature'],
            audio_temperature=self.config['sampling']['audio_temperature'],
            lambda1=self.config['sampling']['cfg_lambda1'],
            lambda2=self.config['sampling']['cfg_lambda2']
        )
        
        # Music structure analyzer
        self.structure_analyzer = MusicStructureAnalyzer(
            self.clap_model, self.rvq_model
        )
        
        logger.info("Models initialized successfully")
    
    def _init_datasets(self):
        """Initialize training and validation datasets"""
        logger.info("Initializing datasets...")
        
        # Audio processor
        self.audio_processor = AudioProcessor(
            sample_rate=self.config['audio']['sample_rate'],
            segment_length=self.config['audio']['segment_length']
        )
        
        # Create dataloaders
        self.train_loader, self.val_loader = create_dataloaders(
            config=self.config,
            clap_model=self.clap_model,
            rvq_model=self.rvq_model,
            audio_processor=self.audio_processor
        )
        
        logger.info(f"Train dataset size: {len(self.train_loader.dataset)}")
        logger.info(f"Validation dataset size: {len(self.val_loader.dataset)}")
    
    def _init_training_components(self):
        """Initialize optimizers, schedulers, and loss functions"""
        logger.info("Initializing training components...")
        
        # Loss function
        self.criterion = MusiCoTLoss(
            ce_weight=self.config['loss']['ce_weight'],
            commitment_weight=self.config['loss']['commitment_weight'],
            diversity_weight=self.config['loss']['diversity_weight']
        )
        
        # Optimizers
        semantic_lm_params = list(self.semantic_lm.parameters())
        rvq_params = list(self.rvq_model.parameters())
        
        self.semantic_optimizer = optim.AdamW(
            semantic_lm_params,
            lr=self.config['training']['lr_semantic'],
            weight_decay=self.config['training']['weight_decay'],
            betas=(0.9, 0.95)
        )
        
        self.rvq_optimizer = optim.AdamW(
            rvq_params,
            lr=self.config['training']['lr_rvq'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate schedulers
        self.semantic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.semantic_optimizer,
            T_max=self.config['training']['num_epochs'],
            eta_min=self.config['training']['lr_semantic'] * 0.01
        )
        
        self.rvq_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.rvq_optimizer,
            T_max=self.config['training']['num_epochs'],
            eta_min=self.config['training']['lr_rvq'] * 0.01
        )
        
        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.config['training']['mixed_precision']
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        logger.info("Training components initialized")
    
    def _init_logging(self):
        """Initialize logging and monitoring"""
        if self.local_rank == 0:  # Only log on main process
            # Tensorboard
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.config['experiment']['log_dir'], 'tensorboard')
            )
            
            # Wandb
            if self.config['experiment']['use_wandb']:
                wandb.init(
                    project=self.config['experiment']['project_name'],
                    name=self.config['experiment']['experiment_name'],
                    config=self.config
                )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.semantic_lm.train()
        self.rvq_model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'ce_loss': 0.0,
            'commitment_loss': 0.0,
            'diversity_loss': 0.0
        }
        
        num_batches = 0
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.epoch + 1}/{self.config['training']['num_epochs']}",
            disable=self.local_rank != 0
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            losses = self._forward_pass(batch)
            
            # Backward pass
            self._backward_pass(losses)
            
            # Update progress
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            num_batches += 1
            
            # Update progress bar
            if self.local_rank == 0:
                progress_bar.set_postfix({
                    'loss': f"{losses['total_loss'].item():.4f}",
                    'lr': f"{self.semantic_optimizer.param_groups[0]['lr']:.2e}"
                })
            
            # Log to tensorboard/wandb
            if self.global_step % self.config['training']['log_interval'] == 0:
                self._log_training_step(losses)
            
            self.global_step += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def _forward_pass(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Forward pass through all models"""
        with torch.cuda.amp.autocast(enabled=self.config['training']['mixed_precision']):
            # Extract CLAP embeddings from audio
            clap_embeddings = self.clap_model.encode_audio(batch['audio'])
            
            # Quantize CLAP embeddings
            quantized_embeddings, rvq_indices, commitment_loss = self.rvq_model(clap_embeddings)
            
            # Forward through semantic LM
            lm_outputs = self.semantic_lm(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                clap_embeddings=batch.get('clap_text_embeddings')
            )
            
            # Compute combined loss
            losses = self.criterion(lm_outputs, commitment_loss, rvq_indices)
            
        return losses
    
    def _backward_pass(self, losses: Dict[str, torch.Tensor]):
        """Backward pass with gradient accumulation"""
        # Scale loss for gradient accumulation
        scaled_loss = losses['total_loss'] / self.config['training']['gradient_accumulation_steps']
        
        # Backward pass
        self.scaler.scale(scaled_loss).backward()
        
        # Update parameters every accumulation_steps
        if (self.global_step + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
            # Clip gradients
            self.scaler.unscale_(self.semantic_optimizer)
            self.scaler.unscale_(self.rvq_optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                self.semantic_lm.parameters(),
                self.config['training']['max_grad_norm']
            )
            torch.nn.utils.clip_grad_norm_(
                self.rvq_model.parameters(),
                self.config['training']['max_grad_norm']
            )
            
            # Optimizer step
            self.scaler.step(self.semantic_optimizer)
            self.scaler.step(self.rvq_optimizer)
            self.scaler.update()
            
            # Zero gradients
            self.semantic_optimizer.zero_grad()
            self.rvq_optimizer.zero_grad()
    
    def validate(self) -> Dict[str, float]:
        """Validation loop"""
        self.semantic_lm.eval()
        self.rvq_model.eval()
        
        val_losses = {
            'total_loss': 0.0,
            'ce_loss': 0.0,
            'commitment_loss': 0.0,
            'diversity_loss': 0.0
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", disable=self.local_rank != 0):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                losses = self._forward_pass(batch)
                
                for key, value in losses.items():
                    val_losses[key] += value.item()
                num_batches += 1
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def _log_training_step(self, losses: Dict[str, torch.Tensor]):
        """Log training metrics"""
        if self.local_rank == 0:
            # Tensorboard
            for key, value in losses.items():
                self.writer.add_scalar(f"train/{key}", value.item(), self.global_step)
            
            self.writer.add_scalar(
                "train/learning_rate", 
                self.semantic_optimizer.param_groups[0]['lr'], 
                self.global_step
            )
            
            # Wandb
            if self.config['experiment']['use_wandb']:
                wandb.log({
                    f"train/{key}": value.item() for key, value in losses.items()
                }, step=self.global_step)
    
    def _log_validation_epoch(self, val_losses: Dict[str, float]):
        """Log validation metrics"""
        if self.local_rank == 0:
            # Tensorboard
            for key, value in val_losses.items():
                self.writer.add_scalar(f"val/{key}", value, self.epoch)
            
            # Wandb
            if self.config['experiment']['use_wandb']:
                wandb.log({
                    f"val/{key}": value for key, value in val_losses.items()
                }, step=self.epoch)
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        if self.local_rank == 0:
            checkpoint = {
                'epoch': self.epoch,
                'global_step': self.global_step,
                'semantic_lm_state_dict': self.semantic_lm.state_dict(),
                'rvq_model_state_dict': self.rvq_model.state_dict(),
                'semantic_optimizer_state_dict': self.semantic_optimizer.state_dict(),
                'rvq_optimizer_state_dict': self.rvq_optimizer.state_dict(),
                'semantic_scheduler_state_dict': self.semantic_scheduler.state_dict(),
                'rvq_scheduler_state_dict': self.rvq_scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'config': self.config
            }
            
            # Save regular checkpoint
            checkpoint_path = os.path.join(
                self.config['experiment']['checkpoint_dir'],
                f"checkpoint_epoch_{self.epoch}.pt"
            )
            torch.save(checkpoint, checkpoint_path)
            
            # Save best checkpoint
            if is_best:
                best_path = os.path.join(
                    self.config['experiment']['checkpoint_dir'],
                    "best_checkpoint.pt"
                )
                torch.save(checkpoint, best_path)
                logger.info(f"Saved best checkpoint at epoch {self.epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.semantic_lm.load_state_dict(checkpoint['semantic_lm_state_dict'])
        self.rvq_model.load_state_dict(checkpoint['rvq_model_state_dict'])
        self.semantic_optimizer.load_state_dict(checkpoint['semantic_optimizer_state_dict'])
        self.rvq_optimizer.load_state_dict(checkpoint['rvq_optimizer_state_dict'])
        self.semantic_scheduler.load_state_dict(checkpoint['semantic_scheduler_state_dict'])
        self.rvq_scheduler.load_state_dict(checkpoint['rvq_scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Resumed training from epoch {self.epoch}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.epoch, self.config['training']['num_epochs']):
            self.epoch = epoch
            
            # Train for one epoch
            train_losses = self.train_epoch()
            
            # Validation
            val_losses = self.validate()
            
            # Update learning rates
            self.semantic_scheduler.step()
            self.rvq_scheduler.step()
            
            # Log validation metrics
            self._log_validation_epoch(val_losses)
            
            # Save checkpoint
            is_best = val_losses['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total_loss']
            
            if epoch % self.config['training']['save_interval'] == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Log epoch summary
            if self.local_rank == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config['training']['num_epochs']} - "
                    f"Train Loss: {train_losses['total_loss']:.4f} - "
                    f"Val Loss: {val_losses['total_loss']:.4f}"
                )
        
        logger.info("Training completed!")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train MusiCoT model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config['local_rank'] = args.local_rank
    
    # Setup distributed training
    if 'WORLD_SIZE' in os.environ:
        config['distributed'] = True
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        config['device'] = f'cuda:{args.local_rank}'
    else:
        config['distributed'] = False
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create output directories
    os.makedirs(config['experiment']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['experiment']['log_dir'], exist_ok=True)
    
    # Initialize trainer
    trainer = MusiCoTTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()