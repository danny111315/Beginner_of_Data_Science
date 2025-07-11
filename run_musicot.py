#!/usr/bin/env python3
"""
MusiCoT Quick Start Script
=========================

This script provides a simple way to get started with MusiCoT training.
It includes data preparation, model initialization, and training launch.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import subprocess
import yaml

def setup_environment():
    """Setup the environment and install dependencies"""
    print("Setting up MusiCoT environment...")
    
    # Install requirements
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies. Please install manually.")
        return False
    
    return True

def create_sample_metadata(output_path: str, num_samples: int = 100):
    """Create sample metadata file for testing"""
    print(f"Creating sample metadata with {num_samples} entries...")
    
    sample_metadata = []
    for i in range(num_samples):
        sample = {
            "audio_file": f"sample_{i:03d}.wav",
            "text_prompt": f"A beautiful song with melody and harmony, sample {i}",
            "lyrics": f"This is sample song number {i}\nWith some example lyrics\nFor testing purposes",
            "genre_tags": ["pop", "electronic"] if i % 2 == 0 else ["rock", "indie"],
            "instrument_tags": ["vocals", "guitar", "drums"] if i % 3 == 0 else ["piano", "bass"],
            "duration": 30.0,
            "artist": f"Sample Artist {i % 10}",
            "title": f"Sample Song {i}"
        }
        sample_metadata.append(sample)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(sample_metadata, f, indent=2)
    
    print(f"✓ Sample metadata saved to {output_path}")

def create_minimal_config(config_path: str):
    """Create a minimal configuration file for quick testing"""
    print("Creating minimal configuration...")
    
    config = {
        "experiment": {
            "project_name": "musicot_quickstart",
            "experiment_name": "test_run",
            "checkpoint_dir": "./checkpoints",
            "log_dir": "./logs",
            "use_wandb": False
        },
        "data": {
            "train_audio_dir": "./data/train_audio",
            "train_metadata_file": "./data/train_metadata.json",
            "val_audio_dir": "./data/val_audio", 
            "val_metadata_file": "./data/val_metadata.json",
            "max_audio_length": 30.0,
            "sample_rate": 44100,
            "segment_length": 30.0,
            "max_text_length": 512,
            "max_lyrics_length": 1024,
            "num_workers": 2,
            "pin_memory": True
        },
        "audio": {
            "sample_rate": 44100,
            "segment_length": 30.0,
            "use_demucs_separation": False  # Disable for quick start
        },
        "clap": {
            "embed_dim": 512,
            "audio_length": 10.0,
            "sample_rate": 48000,
            "text_encoder": "roberta-base",
            "audio_encoder": "HTSAT"
        },
        "rvq": {
            "num_quantizers": 4,  # Reduced for quick start
            "codebook_size": 512,  # Reduced for quick start
            "commitment_weight": 0.25
        },
        "semantic_lm": {
            "model_name": "microsoft/DialoGPT-small",  # Smaller model for testing
            "vocab_size_extension": 5000,  # Reduced
            "max_sequence_length": 1024,  # Reduced
            "use_gradient_checkpointing": True
        },
        "training": {
            "num_epochs": 5,  # Few epochs for testing
            "batch_size": 2,  # Small batch size
            "gradient_accumulation_steps": 2,
            "lr_semantic": 5e-5,
            "lr_rvq": 1e-4,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "mixed_precision": True,
            "log_interval": 10,
            "save_interval": 1,
            "eval_interval": 1
        },
        "loss": {
            "ce_weight": 1.0,
            "commitment_weight": 0.25,
            "diversity_weight": 0.1
        },
        "sampling": {
            "cot_temperature": 0.65,
            "audio_temperature": 0.75,
            "cfg_lambda1": 2.3,
            "cfg_lambda2": 1.3,
            "top_k": 50,
            "top_p": 0.9
        }
    }
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✓ Minimal config saved to {config_path}")

def create_dummy_audio_files(audio_dir: str, num_files: int = 50):
    """Create dummy audio files for testing"""
    print(f"Creating {num_files} dummy audio files...")
    
    try:
        import torch
        import torchaudio
        
        os.makedirs(audio_dir, exist_ok=True)
        
        for i in range(num_files):
            # Generate 30 seconds of dummy audio (sine wave)
            sample_rate = 44100
            duration = 30.0
            num_samples = int(sample_rate * duration)
            
            # Create a simple sine wave with some variation
            frequency = 440 + (i * 10)  # Vary frequency
            t = torch.linspace(0, duration, num_samples)
            waveform = 0.3 * torch.sin(2 * torch.pi * frequency * t)
            
            # Add some harmonics for more interesting sound
            waveform += 0.1 * torch.sin(2 * torch.pi * frequency * 2 * t)
            waveform += 0.05 * torch.sin(2 * torch.pi * frequency * 3 * t)
            
            # Add noise
            waveform += 0.01 * torch.randn_like(waveform)
            
            # Save as wav file
            audio_path = os.path.join(audio_dir, f"sample_{i:03d}.wav")
            torchaudio.save(audio_path, waveform.unsqueeze(0), sample_rate)
        
        print(f"✓ Dummy audio files created in {audio_dir}")
        return True
        
    except ImportError:
        print("✗ PyTorch/torchaudio not available. Cannot create dummy audio files.")
        print("Please provide your own audio files in the specified directory.")
        return False

def setup_data_directories():
    """Setup the data directory structure"""
    print("Setting up data directories...")
    
    directories = [
        "./data/train_audio",
        "./data/val_audio",
        "./checkpoints",
        "./logs",
        "./config"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Data directories created")

def main():
    parser = argparse.ArgumentParser(description="MusiCoT Quick Start")
    parser.add_argument("--setup-only", action="store_true", 
                       help="Only setup environment and data, don't start training")
    parser.add_argument("--skip-install", action="store_true",
                       help="Skip dependency installation")
    parser.add_argument("--config", type=str, default="./config/quickstart_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--num-samples", type=int, default=50,
                       help="Number of dummy samples to create")
    
    args = parser.parse_args()
    
    print("🎵 MusiCoT Quick Start 🎵")
    print("=" * 50)
    
    # Setup environment
    if not args.skip_install:
        if not setup_environment():
            return 1
    
    # Setup directories
    setup_data_directories()
    
    # Create sample data
    create_sample_metadata("./data/train_metadata.json", args.num_samples)
    create_sample_metadata("./data/val_metadata.json", max(args.num_samples // 5, 10))
    
    # Create dummy audio files
    audio_created = create_dummy_audio_files("./data/train_audio", args.num_samples)
    create_dummy_audio_files("./data/val_audio", max(args.num_samples // 5, 10))
    
    # Create minimal config
    create_minimal_config(args.config)
    
    if args.setup_only:
        print("\n✓ Setup completed!")
        print("To start training, run:")
        print(f"python train_musicot.py --config {args.config}")
        return 0
    
    if not audio_created:
        print("\n⚠️  Warning: Dummy audio files could not be created.")
        print("Please provide your own audio files before starting training.")
        return 1
    
    # Start training
    print("\n🚀 Starting MusiCoT training...")
    try:
        subprocess.check_call([
            sys.executable, "train_musicot.py",
            "--config", args.config
        ])
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return 0
    
    print("\n🎉 Training completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())