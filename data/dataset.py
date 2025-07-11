import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchaudio
import numpy as np
import librosa
from transformers import AutoTokenizer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class MusiCoTDataset(Dataset):
    """Dataset class for MusiCoT training"""
    
    def __init__(self,
                 audio_dir: str,
                 metadata_file: str,
                 clap_model,
                 rvq_model,
                 audio_processor,
                 config: Dict,
                 is_training: bool = True):
        """
        Args:
            audio_dir: Directory containing audio files
            metadata_file: JSON file with metadata (lyrics, prompts, etc.)
            clap_model: CLAP model for encoding text/audio
            rvq_model: RVQ model for quantizing CLAP embeddings
            audio_processor: Audio processing utilities
            config: Configuration dictionary
            is_training: Whether this is training dataset
        """
        self.audio_dir = Path(audio_dir)
        self.clap_model = clap_model
        self.rvq_model = rvq_model
        self.audio_processor = audio_processor
        self.config = config
        self.is_training = is_training
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Filter valid samples
        self.samples = self._filter_valid_samples()
        
        # Initialize tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Special tokens for MusiCoT
        self.cot_bos_token = "<cot_bos>"
        self.cot_eos_token = "<cot_eos>"
        self.audio_bos_token = "<audio_bos>"
        self.audio_eos_token = "<audio_eos>"
        self.clap_token = "<clap_token>"
        self.mir_token = "<mir_token>"
        
        print(f"Loaded {len(self.samples)} valid samples for {'training' if is_training else 'validation'}")
    
    def _filter_valid_samples(self) -> List[Dict]:
        """Filter samples that have valid audio files and metadata"""
        valid_samples = []
        
        for sample in self.metadata:
            audio_path = self.audio_dir / sample['audio_file']
            
            # Check if audio file exists
            if not audio_path.exists():
                continue
            
            # Check if required metadata exists
            if not all(key in sample for key in ['text_prompt', 'lyrics']):
                continue
            
            # Check audio duration (optional)
            try:
                info = torchaudio.info(str(audio_path))
                duration = info.num_frames / info.sample_rate
                if duration < 5.0 or duration > self.config['data']['max_audio_length']:
                    continue
            except Exception:
                continue
            
            valid_samples.append(sample)
        
        return valid_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample"""
        sample = self.samples[idx]
        
        try:
            # Load and process audio
            audio_path = self.audio_dir / sample['audio_file']
            audio = self._load_audio(audio_path)
            
            # Extract CLAP embeddings from audio
            clap_audio_embeddings = self._extract_clap_embeddings(audio)
            
            # Quantize CLAP embeddings to get CoT tokens
            rvq_indices = self._quantize_clap_embeddings(clap_audio_embeddings)
            
            # Process text and create training sequence
            training_sequence = self._create_training_sequence(
                sample, rvq_indices, audio
            )
            
            return training_sequence
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return a dummy sample in case of error
            return self._get_dummy_sample()
    
    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """Load and preprocess audio file"""
        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != self.config['audio']['sample_rate']:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.config['audio']['sample_rate']
            )
            waveform = resampler(waveform)
        
        # Trim or pad to desired length
        target_length = int(
            self.config['data']['segment_length'] * self.config['audio']['sample_rate']
        )
        
        if waveform.shape[1] > target_length:
            # Random crop during training, center crop during validation
            if self.is_training:
                start_idx = random.randint(0, waveform.shape[1] - target_length)
            else:
                start_idx = (waveform.shape[1] - target_length) // 2
            waveform = waveform[:, start_idx:start_idx + target_length]
        elif waveform.shape[1] < target_length:
            # Pad with zeros
            pad_amount = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        
        return waveform.squeeze(0)  # Remove channel dimension
    
    def _extract_clap_embeddings(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract CLAP embeddings from audio"""
        # Split audio into 10-second segments for CLAP
        segment_length = int(self.config['clap']['audio_length'] * self.config['audio']['sample_rate'])
        num_segments = len(audio) // segment_length
        
        embeddings = []
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            segment = audio[start_idx:end_idx]
            
            # Extract CLAP embedding
            with torch.no_grad():
                embedding = self.clap_model.encode_audio(segment.unsqueeze(0))
                embeddings.append(embedding.squeeze(0))
        
        if embeddings:
            return torch.stack(embeddings)
        else:
            # Fallback for short audio
            with torch.no_grad():
                embedding = self.clap_model.encode_audio(audio.unsqueeze(0))
                return embedding.squeeze(0).unsqueeze(0)
    
    def _quantize_clap_embeddings(self, clap_embeddings: torch.Tensor) -> torch.Tensor:
        """Quantize CLAP embeddings using RVQ"""
        with torch.no_grad():
            _, indices, _ = self.rvq_model(clap_embeddings.unsqueeze(0))
            return indices.squeeze(0)  # Remove batch dimension
    
    def _create_training_sequence(self, 
                                sample: Dict, 
                                rvq_indices: torch.Tensor,
                                audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create the training sequence for MusiCoT"""
        
        # Extract text components
        text_prompt = sample.get('text_prompt', '')
        lyrics = sample.get('lyrics', '')
        genre_tags = sample.get('genre_tags', [])
        instrument_tags = sample.get('instrument_tags', [])
        
        # Encode text prompt with CLAP
        clap_text_embedding = self._encode_text_with_clap(text_prompt)
        
        # Create MIR tags string
        mir_tags = self._format_mir_tags(genre_tags, instrument_tags)
        
        # Tokenize text components
        text_tokens = self._tokenize_text(text_prompt)
        mir_tokens = self._tokenize_text(mir_tags)
        lyrics_tokens = self._tokenize_text(lyrics)
        
        # Flatten RVQ indices (coarse-to-fine order)
        flattened_rvq = self._flatten_rvq_indices(rvq_indices)
        
        # Create audio tokens (placeholder - would use actual audio tokenizer)
        audio_tokens = self._create_audio_tokens(audio)
        
        # Construct the full training sequence
        sequence_tokens = []
        
        # Add CLAP text embedding placeholder
        sequence_tokens.append(self.tokenizer.convert_tokens_to_ids(self.clap_token))
        
        # Add MIR tags
        sequence_tokens.extend(mir_tokens)
        
        # Add lyrics
        sequence_tokens.extend(lyrics_tokens)
        
        # Add CoT section
        sequence_tokens.append(self.tokenizer.convert_tokens_to_ids(self.cot_bos_token))
        sequence_tokens.extend(flattened_rvq.tolist())
        sequence_tokens.append(self.tokenizer.convert_tokens_to_ids(self.cot_eos_token))
        
        # Add audio section
        sequence_tokens.append(self.tokenizer.convert_tokens_to_ids(self.audio_bos_token))
        sequence_tokens.extend(audio_tokens)
        sequence_tokens.append(self.tokenizer.convert_tokens_to_ids(self.audio_eos_token))
        
        # Convert to tensors
        input_ids = torch.tensor(sequence_tokens[:-1], dtype=torch.long)
        labels = torch.tensor(sequence_tokens[1:], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        # Create the return dictionary
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'clap_text_embeddings': clap_text_embedding,
            'audio': audio,
            'rvq_indices': rvq_indices,
            'metadata': sample
        }
    
    def _encode_text_with_clap(self, text: str) -> torch.Tensor:
        """Encode text using CLAP model"""
        with torch.no_grad():
            embedding = self.clap_model.encode_text([text])
            return embedding.squeeze(0)
    
    def _format_mir_tags(self, genre_tags: List[str], instrument_tags: List[str]) -> str:
        """Format MIR tags into a string"""
        all_tags = []
        if genre_tags:
            all_tags.extend([f"genre:{tag}" for tag in genre_tags])
        if instrument_tags:
            all_tags.extend([f"instrument:{tag}" for tag in instrument_tags])
        return " ".join(all_tags)
    
    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize text using the tokenizer"""
        if not text.strip():
            return []
        
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=self.config['data']['max_text_length'],
            truncation=True
        )
        return tokens
    
    def _flatten_rvq_indices(self, rvq_indices: torch.Tensor) -> torch.Tensor:
        """Flatten RVQ indices in coarse-to-fine order"""
        # rvq_indices shape: [time_steps, num_quantizers]
        time_steps, num_quantizers = rvq_indices.shape
        
        # Transpose and flatten: [num_quantizers, time_steps] -> [num_quantizers * time_steps]
        flattened = rvq_indices.transpose(0, 1).flatten()
        
        # Offset indices to avoid collision with text tokens
        # Assuming text tokens are in range [0, 30000], use [30000, 40000] for RVQ
        offset = 30000
        flattened = flattened + offset
        
        return flattened
    
    def _create_audio_tokens(self, audio: torch.Tensor) -> List[int]:
        """Create audio tokens (placeholder implementation)"""
        # This would use an actual audio tokenizer like BEST-RQ
        # For now, create dummy tokens
        num_tokens = len(audio) // 320  # Assuming 25Hz frame rate
        
        # Use token range [40000, 50000] for audio tokens
        offset = 40000
        tokens = [offset + (i % 1000) for i in range(min(num_tokens, 1000))]
        
        return tokens
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return a dummy sample in case of error"""
        dummy_length = 512
        return {
            'input_ids': torch.zeros(dummy_length, dtype=torch.long),
            'labels': torch.zeros(dummy_length, dtype=torch.long),
            'attention_mask': torch.ones(dummy_length, dtype=torch.long),
            'clap_text_embeddings': torch.zeros(512),
            'audio': torch.zeros(self.config['audio']['sample_rate'] * 30),
            'rvq_indices': torch.zeros(3, 8, dtype=torch.long),
            'metadata': {'text_prompt': '', 'lyrics': '', 'audio_file': 'dummy.wav'}
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching"""
    
    # Get maximum sequence length in the batch
    max_length = max(len(sample['input_ids']) for sample in batch)
    
    # Initialize batch tensors
    batch_size = len(batch)
    
    input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)  # -100 for padding
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)
    
    clap_embeddings = torch.stack([sample['clap_text_embeddings'] for sample in batch])
    
    # Pad sequences
    for i, sample in enumerate(batch):
        seq_len = len(sample['input_ids'])
        input_ids[i, :seq_len] = sample['input_ids']
        labels[i, :seq_len] = sample['labels']
        attention_mask[i, :seq_len] = sample['attention_mask']
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'clap_text_embeddings': clap_embeddings,
        'audio': torch.stack([sample['audio'] for sample in batch]),
        'metadata': [sample['metadata'] for sample in batch]
    }


def create_dataloaders(config: Dict, 
                      clap_model, 
                      rvq_model, 
                      audio_processor) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders"""
    
    # Training dataset
    train_dataset = MusiCoTDataset(
        audio_dir=config['data']['train_audio_dir'],
        metadata_file=config['data']['train_metadata_file'],
        clap_model=clap_model,
        rvq_model=rvq_model,
        audio_processor=audio_processor,
        config=config,
        is_training=True
    )
    
    # Validation dataset
    val_dataset = MusiCoTDataset(
        audio_dir=config['data']['val_audio_dir'],
        metadata_file=config['data']['val_metadata_file'],
        clap_model=clap_model,
        rvq_model=rvq_model,
        audio_processor=audio_processor,
        config=config,
        is_training=False
    )
    
    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    
    if config.get('distributed', False):
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_fn,
        drop_last=False
    )
    
    return train_loader, val_loader


class AudioProcessor:
    """Audio processing utilities"""
    
    def __init__(self, sample_rate: int = 44100, segment_length: float = 30.0):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file and preprocess"""
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0)
    
    def extract_features(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract audio features"""
        # Convert to numpy for librosa
        audio_np = audio.numpy()
        
        # Extract features
        features = {}
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_np,
            sr=self.sample_rate,
            n_mels=128,
            hop_length=512
        )
        features['mel_spectrogram'] = torch.from_numpy(mel_spec)
        
        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio_np,
            sr=self.sample_rate,
            n_mfcc=13
        )
        features['mfcc'] = torch.from_numpy(mfcc)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(
            y=audio_np,
            sr=self.sample_rate
        )
        features['chroma'] = torch.from_numpy(chroma)
        
        return features