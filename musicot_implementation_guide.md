# MusiCoT Implementation Guide

## Overview
This guide provides a comprehensive implementation plan for reproducing the MusiCoT (Chain-of-Musical-Thought) experiment for high-fidelity music generation.

## System Architecture

The MusiCoT system consists of several key components:
1. **CLAP Model**: Contrastive Language-Audio Pretraining model
2. **RVQ Model**: Residual Vector Quantization for CLAP embeddings
3. **Semantic Language Model**: LLaMA-based model for token prediction
4. **Diffusion Model**: Stable Audio-based acoustic model
5. **Audio VAE-GAN**: For high-quality waveform reconstruction

## Implementation Steps

### 1. Environment Setup

```bash
# Create conda environment
conda create -n musicot python=3.9
conda activate musicot

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers accelerate
pip install librosa soundfile
pip install diffusers
pip install vector-quantize-pytorch
pip install laion-clap
```

### 2. Data Preparation

#### Required Datasets:
- **DISCO-10M**: 10 million English songs
- **Additional music tracks**: ~200,000 in-house tracks
- **Evaluation data**: 100 generated lyrics using ChatGPT

#### Data Processing Pipeline:
1. **Source Separation**: Use Demucs to extract vocals
2. **ASR Transcription**: Extract lyrics with timestamps
3. **VAD**: Voice Activity Detection for silent segments
4. **Music Segmentation**: Use All-in-One model for structure analysis

### 3. Model Components Implementation

#### 3.1 CLAP Model Training
```python
# Based on LAION-CLAP implementation
from clap import CLAP

def train_clap_model():
    model = CLAP(
        audio_encoder='HTSAT',
        text_encoder='roberta',
        embed_dim=512,
        audio_length=10.0,  # 10-second segments
        sample_rate=48000
    )
    
    # Training configuration
    config = {
        'batch_size': 64,
        'learning_rate': 1e-4,
        'epochs': 100,
        'temperature': 0.07
    }
    
    return model, config
```

#### 3.2 RVQ Model for CLAP Embeddings
```python
import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ

class CLAPQuantizer(nn.Module):
    def __init__(self, embed_dim=512, num_quantizers=8, codebook_size=1024):
        super().__init__()
        self.rvq = ResidualVQ(
            dim=embed_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            kmeans_init=True,
            kmeans_iters=10
        )
        
    def forward(self, clap_embeddings):
        # clap_embeddings: [batch, time, embed_dim]
        quantized, indices, _ = self.rvq(clap_embeddings)
        return quantized, indices
    
    def encode(self, clap_embeddings):
        _, indices, _ = self.rvq(clap_embeddings)
        return indices  # [batch, time, num_quantizers]
```

#### 3.3 Semantic Language Model (LLaMA-based)
```python
from transformers import LlamaForCausalLM, LlamaTokenizer

class MusiCoTSemanticLM(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf"):
        super().__init__()
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name)
        
        # Add special tokens for MusiCoT
        special_tokens = ["<cot_bos>", "<cot_eos>", "<audio_bos>", "<audio_eos>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
```

#### 3.4 Training Data Format
```python
class MusiCoTDataset(torch.utils.data.Dataset):
    def __init__(self, audio_files, lyrics, text_prompts, clap_model, rvq_model):
        self.audio_files = audio_files
        self.lyrics = lyrics
        self.text_prompts = text_prompts
        self.clap_model = clap_model
        self.rvq_model = rvq_model
        
    def __getitem__(self, idx):
        # Load audio and extract CLAP embeddings
        audio = self.load_audio(self.audio_files[idx])
        clap_embeddings = self.clap_model.encode_audio(audio)
        
        # Quantize CLAP embeddings
        rvq_tokens = self.rvq_model.encode(clap_embeddings)
        
        # Create training sequence
        sequence = self.create_training_sequence(
            self.text_prompts[idx],
            self.lyrics[idx],
            rvq_tokens,
            audio
        )
        
        return sequence
    
    def create_training_sequence(self, text_prompt, lyrics, rvq_tokens, audio):
        # Format: [CLAP_text] + [MIR_tags] + [lyrics] + <cot_bos> + [RVQ_tokens] + <cot_eos> + [audio_tokens]
        sequence = {
            'clap_embedding': self.clap_model.encode_text(text_prompt),
            'mir_tags': self.extract_mir_tags(audio),
            'lyrics_tokens': self.tokenize_lyrics(lyrics),
            'rvq_tokens': rvq_tokens.flatten(),  # Flatten coarse-to-fine
            'audio_tokens': self.extract_audio_tokens(audio)
        }
        return sequence
```

### 4. Training Pipeline

#### 4.1 MusiCoT Training
```python
def train_musicot():
    # Initialize models
    clap_model, _ = train_clap_model()
    rvq_model = CLAPQuantizer()
    semantic_lm = MusiCoTSemanticLM()
    
    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Forward pass
            outputs = semantic_lm(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            # Compute loss
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

#### 4.2 Dual-Sampling Strategies
```python
class DualSamplingStrategy:
    def __init__(self, cot_temperature=0.65, audio_temperature=0.75, 
                 lambda1=2.3, lambda2=1.3):
        self.cot_temp = cot_temperature
        self.audio_temp = audio_temperature
        self.lambda1 = lambda1
        self.lambda2 = lambda2
    
    def dual_temperature_sample(self, logits, token_type):
        if token_type == "cot":
            logits = logits / self.cot_temp
        else:  # audio tokens
            logits = logits / self.audio_temp
        
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)
    
    def dual_scale_cfg(self, cond_logits, uncond_logits, token_type):
        if token_type == "cot":
            lambda_val = self.lambda1
        else:
            lambda_val = self.lambda2
        
        return lambda_val * cond_logits + (1 - lambda_val) * uncond_logits
```

### 5. Inference Pipeline

```python
def generate_music_with_musicot(text_prompt, lyrics=None, reference_audio=None):
    # Step 1: Generate MusiCoT tokens (CLAP RVQ)
    cot_tokens = generate_cot_tokens(text_prompt, lyrics)
    
    # Step 2: Generate semantic audio tokens
    audio_tokens = generate_audio_tokens(cot_tokens, text_prompt, lyrics)
    
    # Step 3: Convert to waveform using diffusion model
    waveform = diffusion_model.decode(audio_tokens)
    
    return waveform, cot_tokens  # Return both audio and analyzable thoughts

def analyze_musical_structure(cot_tokens, text_anchors=None):
    """Analyze instrumental arrangements using CLAP RVQ tokens"""
    if text_anchors is None:
        text_anchors = ['vocals', 'bass', 'drums', 'guitar', 'piano']
    
    # Convert RVQ tokens back to CLAP embeddings
    clap_embeddings = rvq_model.decode(cot_tokens)
    
    # Compute similarities with text anchors
    analysis = {}
    for anchor in text_anchors:
        anchor_embedding = clap_model.encode_text(anchor)
        similarity = cosine_similarity(clap_embeddings, anchor_embedding)
        analysis[anchor] = similarity
    
    return analysis
```

### 6. Evaluation Metrics

```python
def evaluate_musicot(generated_audio, reference_audio):
    metrics = {}
    
    # FAD using CLAP
    metrics['fad'] = compute_clap_fad(generated_audio, reference_audio)
    
    # Content scores using Meta Audiobox-Aesthetic
    content_scores = compute_content_scores(generated_audio)
    metrics.update(content_scores)
    
    # Real-time factor
    metrics['rtf'] = generation_time / audio_duration
    
    return metrics

def compute_clap_fad(gen_audio, ref_audio):
    """Compute FAD using CLAP embeddings at 48kHz stereo"""
    gen_embeddings = clap_model.encode_audio(gen_audio)
    ref_embeddings = clap_model.encode_audio(ref_audio)
    
    # Compute Fréchet distance
    mu1, sigma1 = gen_embeddings.mean(0), np.cov(gen_embeddings.T)
    mu2, sigma2 = ref_embeddings.mean(0), np.cov(ref_embeddings.T)
    
    fad = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fad
```

### 7. Key Implementation Notes

1. **Model Scaling**: Use ~1B parameter models for both semantic LM and diffusion model
2. **Data Processing**: Implement robust audio preprocessing pipeline with Demucs separation
3. **Training Efficiency**: Use gradient accumulation and mixed precision training
4. **Memory Management**: Implement checkpointing for large model training
5. **Evaluation**: Use professional musicians for subjective evaluation (MOS)

### 8. Experimental Configuration

```yaml
# config.yaml
model:
  semantic_lm:
    architecture: "llama"
    parameters: 1e9
    context_length: 4096
  
  diffusion:
    architecture: "stable_audio"
    parameters: 1e9
    sample_rate: 44100
  
  clap:
    embed_dim: 512
    audio_length: 10.0
    sample_rate: 48000

training:
  batch_size: 32
  learning_rate: 1e-4
  epochs: 100
  gradient_accumulation_steps: 4

sampling:
  cot_temperature: 0.65
  audio_temperature: 0.75
  cfg_lambda1: 2.3
  cfg_lambda2: 1.3
```

## Expected Results

- **MOS Improvement**: From 3.35 to 3.72
- **FAD Improvement**: From 0.112 to 0.102
- **Structural Analyzability**: Correlation coefficients > 0.5 for instrument detection
- **Real-time Factor**: ~0.27 (faster than real-time)

## Next Steps

1. Start with CLAP model training on your music dataset
2. Implement and train the RVQ quantizer
3. Fine-tune the semantic language model with MusiCoT objectives
4. Train the diffusion model for acoustic generation
5. Implement dual-sampling strategies
6. Conduct comprehensive evaluation

This implementation provides a solid foundation for reproducing the MusiCoT experiment. The modular design allows for iterative development and testing of individual components.