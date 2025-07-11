import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from vector_quantize_pytorch import ResidualVQ
import numpy as np
from typing import Optional, Dict, List, Tuple

class CLAPQuantizer(nn.Module):
    """Residual Vector Quantization for CLAP embeddings"""
    
    def __init__(self, 
                 embed_dim: int = 512, 
                 num_quantizers: int = 8, 
                 codebook_size: int = 1024,
                 commitment_weight: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        
        self.rvq = ResidualVQ(
            dim=embed_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            kmeans_init=True,
            kmeans_iters=10,
            commitment_weight=commitment_weight
        )
        
    def forward(self, clap_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            clap_embeddings: [batch, time, embed_dim]
        Returns:
            quantized: [batch, time, embed_dim]
            indices: [batch, time, num_quantizers]
            commit_loss: scalar
        """
        quantized, indices, commit_loss = self.rvq(clap_embeddings)
        return quantized, indices, commit_loss
    
    def encode(self, clap_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode CLAP embeddings to quantized indices"""
        _, indices, _ = self.rvq(clap_embeddings)
        return indices  # [batch, time, num_quantizers]
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode quantized indices back to embeddings"""
        return self.rvq.get_output_from_indices(indices)
    
    def flatten_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Flatten indices from coarse-to-fine order"""
        batch_size, time_steps, num_quantizers = indices.shape
        # Reshape to [batch, time_steps * num_quantizers]
        flattened = indices.transpose(1, 2).reshape(batch_size, -1)
        return flattened


class MusiCoTSemanticLM(nn.Module):
    """Semantic Language Model with MusiCoT capabilities"""
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-hf",
                 vocab_size_extension: int = 10000,  # For audio and CoT tokens
                 max_sequence_length: int = 4096):
        super().__init__()
        
        # Load base LLaMA model
        self.config = LlamaConfig.from_pretrained(model_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name)
        
        # Add special tokens for MusiCoT
        special_tokens = {
            "additional_special_tokens": [
                "<cot_bos>", "<cot_eos>", 
                "<audio_bos>", "<audio_eos>",
                "<clap_token>", "<mir_token>"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Extend vocabulary for audio and CoT tokens
        original_vocab_size = len(self.tokenizer)
        new_vocab_size = original_vocab_size + vocab_size_extension
        self.model.resize_token_embeddings(new_vocab_size)
        
        # Store token type ranges
        self.text_token_range = (0, original_vocab_size)
        self.cot_token_range = (original_vocab_size, original_vocab_size + 8192)  # 8192 CoT tokens
        self.audio_token_range = (original_vocab_size + 8192, new_vocab_size)
        
        # Special token IDs
        self.cot_bos_id = self.tokenizer.convert_tokens_to_ids("<cot_bos>")
        self.cot_eos_id = self.tokenizer.convert_tokens_to_ids("<cot_eos>")
        self.audio_bos_id = self.tokenizer.convert_tokens_to_ids("<audio_bos>")
        self.audio_eos_id = self.tokenizer.convert_tokens_to_ids("<audio_eos>")
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                clap_embeddings: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass with optional CLAP embeddings injection
        """
        # Inject CLAP embeddings if provided
        if clap_embeddings is not None:
            # Replace CLAP token positions with actual embeddings
            embeddings = self.model.get_input_embeddings()(input_ids)
            clap_mask = (input_ids == self.tokenizer.convert_tokens_to_ids("<clap_token>"))
            embeddings[clap_mask] = clap_embeddings
            
            outputs = self.model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                labels=labels
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        
        return outputs
    
    def get_token_type(self, token_id: int) -> str:
        """Determine the type of token (text, cot, audio)"""
        if self.text_token_range[0] <= token_id < self.text_token_range[1]:
            return "text"
        elif self.cot_token_range[0] <= token_id < self.cot_token_range[1]:
            return "cot"
        elif self.audio_token_range[0] <= token_id < self.audio_token_range[1]:
            return "audio"
        else:
            return "unknown"


class DualSamplingStrategy:
    """Dual-temperature and dual-scale CFG sampling strategies"""
    
    def __init__(self, 
                 cot_temperature: float = 0.65,
                 audio_temperature: float = 0.75,
                 lambda1: float = 2.3,  # CFG scale for CoT tokens
                 lambda2: float = 1.3):  # CFG scale for audio tokens
        self.cot_temp = cot_temperature
        self.audio_temp = audio_temperature
        self.lambda1 = lambda1
        self.lambda2 = lambda2
    
    def apply_dual_temperature(self, logits: torch.Tensor, token_type: str) -> torch.Tensor:
        """Apply temperature based on token type"""
        if token_type == "cot":
            return logits / self.cot_temp
        elif token_type == "audio":
            return logits / self.audio_temp
        else:
            return logits  # No temperature for text tokens
    
    def apply_dual_scale_cfg(self, 
                           cond_logits: torch.Tensor, 
                           uncond_logits: torch.Tensor, 
                           token_type: str) -> torch.Tensor:
        """Apply classifier-free guidance with dual scales"""
        if token_type == "cot":
            lambda_val = self.lambda1
        elif token_type == "audio":
            lambda_val = self.lambda2
        else:
            lambda_val = 1.0  # No CFG for text tokens
        
        return lambda_val * cond_logits + (1 - lambda_val) * uncond_logits
    
    def sample(self, 
               cond_logits: torch.Tensor, 
               uncond_logits: torch.Tensor, 
               token_type: str,
               top_k: int = 50,
               top_p: float = 0.9) -> torch.Tensor:
        """Complete sampling with dual strategies"""
        # Apply dual-scale CFG
        logits = self.apply_dual_scale_cfg(cond_logits, uncond_logits, token_type)
        
        # Apply dual temperature
        logits = self.apply_dual_temperature(logits, token_type)
        
        # Apply top-k and top-p filtering
        logits = self._top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        
        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    def _top_k_top_p_filtering(self, 
                              logits: torch.Tensor, 
                              top_k: int = 0, 
                              top_p: float = 1.0) -> torch.Tensor:
        """Filter logits using top-k and top-p (nucleus) sampling"""
        if top_k > 0:
            # Get top-k indices
            top_k_values, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
            # Set all other values to -inf
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(-1, top_k_indices, top_k_values)
            logits = logits_filtered
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter the boolean mask back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        return logits


class MusicStructureAnalyzer:
    """Analyze musical structure using CLAP RVQ tokens"""
    
    def __init__(self, clap_model, rvq_model):
        self.clap_model = clap_model
        self.rvq_model = rvq_model
        
        # Predefined text anchors for analysis
        self.instrument_anchors = [
            'vocals', 'singing', 'voice',
            'bass', 'bass guitar', 'bass line',
            'drums', 'percussion', 'beat',
            'guitar', 'electric guitar', 'acoustic guitar',
            'piano', 'keyboard', 'keys',
            'violin', 'strings', 'orchestra',
            'trumpet', 'brass', 'horn',
            'synthesizer', 'synth', 'electronic'
        ]
        
        self.mood_anchors = [
            'happy', 'energetic', 'upbeat',
            'sad', 'melancholic', 'dark',
            'calm', 'peaceful', 'relaxing',
            'aggressive', 'intense', 'powerful'
        ]
        
        self.genre_anchors = [
            'rock', 'pop', 'jazz', 'classical',
            'electronic', 'hip hop', 'country',
            'blues', 'reggae', 'folk'
        ]
    
    def analyze_structure(self, 
                         cot_tokens: torch.Tensor, 
                         anchor_type: str = "instruments") -> Dict[str, np.ndarray]:
        """
        Analyze musical structure using CoT tokens
        
        Args:
            cot_tokens: [time_steps, num_quantizers] or flattened
            anchor_type: "instruments", "moods", or "genres"
        
        Returns:
            Dictionary with anchor names as keys and similarity arrays as values
        """
        # Get appropriate anchors
        if anchor_type == "instruments":
            anchors = self.instrument_anchors
        elif anchor_type == "moods":
            anchors = self.mood_anchors
        elif anchor_type == "genres":
            anchors = self.genre_anchors
        else:
            raise ValueError(f"Unknown anchor type: {anchor_type}")
        
        # Convert RVQ tokens back to CLAP embeddings
        if len(cot_tokens.shape) == 1:
            # Reshape flattened tokens
            time_steps = cot_tokens.shape[0] // self.rvq_model.num_quantizers
            cot_tokens = cot_tokens.reshape(time_steps, self.rvq_model.num_quantizers)
        
        clap_embeddings = self.rvq_model.decode(cot_tokens.unsqueeze(0))  # Add batch dim
        clap_embeddings = clap_embeddings.squeeze(0)  # Remove batch dim
        
        # Compute similarities with each anchor
        analysis_results = {}
        for anchor in anchors:
            anchor_embedding = self.clap_model.encode_text([anchor])
            
            # Compute cosine similarity for each time step
            similarities = []
            for t in range(clap_embeddings.shape[0]):
                sim = F.cosine_similarity(
                    clap_embeddings[t:t+1], 
                    anchor_embedding, 
                    dim=-1
                )
                similarities.append(sim.item())
            
            analysis_results[anchor] = np.array(similarities)
        
        return analysis_results
    
    def visualize_analysis(self, analysis_results: Dict[str, np.ndarray], 
                          time_duration: float = 30.0):
        """Create visualization of the analysis results"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            time_steps = len(list(analysis_results.values())[0])
            time_axis = np.linspace(0, time_duration, time_steps)
            
            for anchor, similarities in analysis_results.items():
                ax.plot(time_axis, similarities, label=anchor, linewidth=2)
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Similarity Score')
            ax.set_title('Musical Structure Analysis')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            print("Matplotlib not available for visualization")
            return None


class MusiCoTLoss(nn.Module):
    """Combined loss function for MusiCoT training"""
    
    def __init__(self, 
                 ce_weight: float = 1.0,
                 commitment_weight: float = 0.25,
                 diversity_weight: float = 0.1):
        super().__init__()
        self.ce_weight = ce_weight
        self.commitment_weight = commitment_weight
        self.diversity_weight = diversity_weight
        
    def forward(self, 
                lm_outputs: Dict,
                rvq_commitment_loss: torch.Tensor,
                rvq_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            lm_outputs: Outputs from language model
            rvq_commitment_loss: Commitment loss from RVQ
            rvq_indices: RVQ token indices for diversity calculation
        """
        losses = {}
        
        # Cross-entropy loss from language model
        if lm_outputs.loss is not None:
            losses['ce_loss'] = lm_outputs.loss * self.ce_weight
        
        # RVQ commitment loss
        losses['commitment_loss'] = rvq_commitment_loss * self.commitment_weight
        
        # Diversity loss to encourage diverse token usage
        if self.diversity_weight > 0:
            diversity_loss = self._compute_diversity_loss(rvq_indices)
            losses['diversity_loss'] = diversity_loss * self.diversity_weight
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_diversity_loss(self, rvq_indices: torch.Tensor) -> torch.Tensor:
        """Encourage diverse usage of codebook tokens"""
        batch_size, seq_len, num_quantizers = rvq_indices.shape
        
        diversity_losses = []
        for q in range(num_quantizers):
            # Get token distribution for this quantizer
            tokens = rvq_indices[:, :, q].flatten()
            unique_tokens, counts = torch.unique(tokens, return_counts=True)
            
            # Compute entropy to encourage diversity
            probs = counts.float() / counts.sum()
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            
            # Convert to loss (negative entropy)
            diversity_losses.append(-entropy)
        
        return torch.stack(diversity_losses).mean()