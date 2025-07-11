# MusiCoT: Analyzable Chain-of-Musical-Thought Prompting

This repository provides a complete implementation of MusiCoT (Chain-of-Musical-Thought) for high-fidelity music generation, based on the research paper "Analyzable Chain-of-Musical-Thought Prompting for High-Fidelity Music Generation".

## 🎵 Overview

MusiCoT is an innovative approach that enhances autoregressive music generation models by introducing a chain-of-thought (CoT) prompting technique specifically designed for music creation. The key innovation is using CLAP (Contrastive Language-Audio Pretraining) embeddings as "musical thoughts" rather than natural language, making the approach scalable and independent of human-labeled data.

### Key Features

- **🎼 Structural Analyzability**: Analyze musical elements like instrumental arrangements
- **🔄 Music Referencing**: Support for variable-length audio inputs as style references
- **📈 Superior Performance**: Improved MOS scores and FAD metrics
- **⚡ Efficient Training**: No additional inference time overhead

## 🏗️ Architecture

The MusiCoT system consists of several key components:

1. **CLAP Model**: Encodes text and audio into a shared embedding space
2. **RVQ Model**: Residual Vector Quantization for CLAP embeddings
3. **Semantic Language Model**: LLaMA-based model with MusiCoT capabilities
4. **Diffusion Model**: For high-quality acoustic generation
5. **Dual-Sampling Strategy**: Temperature and CFG control for different token types

## 🚀 Quick Start

### Option 1: Automated Setup
```bash
# Clone the repository (if applicable)
git clone <your-repo-url>
cd musicot

# Quick start with automatic setup
python run_musicot.py
```

### Option 2: Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup data directories
mkdir -p data/{train_audio,val_audio} checkpoints logs config

# 3. Prepare your data (see Data Preparation section)

# 4. Configure training
cp config/musicot_config.yaml config/my_config.yaml
# Edit config/my_config.yaml with your data paths

# 5. Start training
python train_musicot.py --config config/my_config.yaml
```

## 📊 Data Preparation

### Audio Data
Your audio data should be organized as follows:
```
data/
├── train_audio/
│   ├── song001.wav
│   ├── song002.wav
│   └── ...
├── val_audio/
│   ├── val001.wav
│   └── ...
├── train_metadata.json
└── val_metadata.json
```

### Metadata Format
The metadata JSON files should contain the following structure:
```json
[
  {
    "audio_file": "song001.wav",
    "text_prompt": "An upbeat pop song with electronic elements",
    "lyrics": "Verse 1 lyrics\nChorus lyrics\nVerse 2 lyrics",
    "genre_tags": ["pop", "electronic"],
    "instrument_tags": ["vocals", "guitar", "drums", "synth"],
    "duration": 180.5,
    "artist": "Artist Name",
    "title": "Song Title"
  }
]
```

### Data Requirements
- **Audio Format**: WAV, MP3, or FLAC
- **Sample Rate**: 44.1kHz (will be resampled if different)
- **Duration**: 5-180 seconds per track
- **Quality**: High-quality music recordings recommended

## ⚙️ Configuration

The configuration file (`config/musicot_config.yaml`) contains all training parameters. Key sections include:

### Model Configuration
```yaml
semantic_lm:
  model_name: "meta-llama/Llama-2-7b-hf"  # Base language model
  vocab_size_extension: 10000             # Additional tokens for audio/CoT
  max_sequence_length: 4096               # Context length

clap:
  embed_dim: 512                          # CLAP embedding dimension
  audio_length: 10.0                      # Segment length for CLAP

rvq:
  num_quantizers: 8                       # Number of quantization levels
  codebook_size: 1024                     # Codebook size per level
```

### Training Configuration
```yaml
training:
  num_epochs: 100
  batch_size: 8
  gradient_accumulation_steps: 4
  lr_semantic: 1e-4                       # Learning rate for semantic LM
  lr_rvq: 5e-4                           # Learning rate for RVQ
  mixed_precision: true                   # Enable mixed precision training
```

### Sampling Configuration
```yaml
sampling:
  cot_temperature: 0.65                   # Temperature for CoT tokens
  audio_temperature: 0.75                 # Temperature for audio tokens
  cfg_lambda1: 2.3                        # CFG scale for CoT
  cfg_lambda2: 1.3                        # CFG scale for audio
```

## 🔧 Training

### Single GPU Training
```bash
python train_musicot.py --config config/musicot_config.yaml
```

### Multi-GPU Training
```bash
# Using torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_musicot.py --config config/musicot_config.yaml

# Using accelerate
accelerate launch train_musicot.py --config config/musicot_config.yaml
```

### Resume Training
```bash
python train_musicot.py \
    --config config/musicot_config.yaml \
    --resume checkpoints/checkpoint_epoch_10.pt
```

## 🎯 Inference and Generation

### Basic Music Generation
```python
from models.musicot_models import MusiCoTSemanticLM, DualSamplingStrategy
from utils.clap_model import CLAPModel

# Load trained models
semantic_lm = MusiCoTSemanticLM.from_pretrained("path/to/checkpoint")
clap_model = CLAPModel.from_pretrained("path/to/clap")

# Generate music
text_prompt = "An energetic rock song with guitar solos"
lyrics = "Rock and roll all night\nParty every day"

waveform, cot_tokens = generate_music_with_musicot(
    text_prompt=text_prompt,
    lyrics=lyrics,
    model=semantic_lm
)
```

### Music Structure Analysis
```python
from models.musicot_models import MusicStructureAnalyzer

# Analyze generated music structure
analyzer = MusicStructureAnalyzer(clap_model, rvq_model)
analysis = analyzer.analyze_structure(cot_tokens, anchor_type="instruments")

# Visualize analysis
fig = analyzer.visualize_analysis(analysis, time_duration=30.0)
fig.show()
```

### Music Referencing
```python
# Generate music with reference audio
reference_audio = load_audio("reference_song.wav")
waveform, cot_tokens = generate_music_with_musicot(
    text_prompt="A song in similar style",
    reference_audio=reference_audio,
    model=semantic_lm
)
```

## 📈 Evaluation

### Objective Metrics
```python
from utils.evaluation import MusiCoTEvaluator

evaluator = MusiCoTEvaluator()
metrics = evaluator.evaluate(
    generated_audio=waveform,
    reference_audio=reference_audio
)

print(f"FAD Score: {metrics['fad']:.3f}")
print(f"CLAP Score: {metrics['clap_score']:.3f}")
```

### Supported Metrics
- **FAD (Fréchet Audio Distance)**: Using CLAP embeddings at 48kHz
- **CLAP Score**: Semantic similarity between text and audio
- **Content Scores**: Production quality assessment
- **MOS (Mean Opinion Score)**: Subjective quality rating

## 🛠️ Advanced Usage

### Custom CLAP Model
```python
# Train your own CLAP model
from utils.clap_model import CLAPModel

clap_model = CLAPModel(
    embed_dim=512,
    audio_encoder="HTSAT",
    text_encoder="roberta-base"
)

# Fine-tune on your music dataset
clap_model.train(your_music_dataset)
```

### Custom Audio Tokenizer
```python
# Replace the placeholder audio tokenizer
from your_tokenizer import BESTRQTokenizer

audio_tokenizer = BESTRQTokenizer(
    sample_rate=44100,
    frame_rate=25
)

# Update dataset to use your tokenizer
dataset.audio_tokenizer = audio_tokenizer
```

### Distributed Training with DeepSpeed
```yaml
# Add to config
training:
  use_deepspeed: true
  deepspeed_config: "config/deepspeed_config.json"
```

## 📊 Expected Results

Based on the paper, you should expect:

- **MOS Improvement**: From 3.35 to 3.72
- **FAD Improvement**: From 0.112 to 0.102  
- **Real-time Factor**: ~0.27 (faster than real-time)
- **Structural Correlation**: >0.5 for instrument detection

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```yaml
   # Reduce batch size and increase gradient accumulation
   training:
     batch_size: 4
     gradient_accumulation_steps: 8
   ```

2. **Slow Data Loading**
   ```yaml
   data:
     num_workers: 0  # Try reducing if experiencing issues
     pin_memory: false
   ```

3. **Model Loading Errors**
   ```bash
   # Install transformers from source for latest LLaMA support
   pip install git+https://github.com/huggingface/transformers
   ```

### Performance Optimization

- Use mixed precision training: `mixed_precision: true`
- Enable gradient checkpointing for large models
- Use efficient data loading with multiple workers
- Consider model compilation with PyTorch 2.0: `compile_model: true`

## 📚 Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{musicot2024,
  title={Analyzable Chain-of-Musical-Thought Prompting for High-Fidelity Music Generation},
  author={Lam, Max W. Y. and Xing, Yijin and You, Weiya and others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:

- Code style and formatting
- Adding new features
- Reporting bugs
- Improving documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original MusiCoT paper authors
- LAION team for CLAP model
- Stability AI for Stable Audio
- Facebook Research for Demucs
- Meta AI for LLaMA models

## 🔗 Resources

- [Paper (arXiv)](https://musicot.github.io/)
- [Demo Samples](https://musicot.github.io/)
- [CLAP Model](https://github.com/LAION-AI/CLAP)
- [Stable Audio](https://github.com/Stability-AI/stable-audio-tools)
- [Vector Quantization](https://github.com/lucidrains/vector-quantize-pytorch)

---

For questions and support, please open an issue or contact the maintainers.