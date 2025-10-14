"""
Hybrid Transformer Model combining LM2's internal memory with LongMem's external retrieval.
Generic implementation that works with any transformer architecture (Llama, Qwen, GPT, etc.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from .internal_memory import InternalMemoryModule
from .external_memory import ExternalMemoryBank


class HybridTransformerConfig(PretrainedConfig):
    """Extended config with hybrid memory parameters for any transformer model."""
    
    model_type = "hybrid_transformer"
    
    def __init__(self, **kwargs):
        # Extract memory-specific parameters before calling super
        self.use_internal_memory = kwargs.pop("use_internal_memory", True)
        
        # Dynamic memory slot calculation based on hidden_dim
        hidden_dim = kwargs.get("hidden_size", 2048)
        default_memory_slots = hidden_dim // 128
        self.memory_slots = kwargs.pop("memory_slots", default_memory_slots)
        
        # Dynamic memory heads calculation
        default_num_mem_heads = max(1, self.memory_slots // 4)
        self.num_mem_heads = kwargs.pop("num_mem_heads", default_num_mem_heads)
        
        self.use_external_memory = kwargs.pop("use_external_memory", True)
        self.external_memory_size = kwargs.pop("external_memory_size", 1048576)
        self.retrieval_k = kwargs.pop("retrieval_k", 8)
        self.chunk_size = kwargs.pop("chunk_size", 4)
        self.use_gpu_to_search = kwargs.pop("use_gpu_to_search", True)
        self.batch_size = kwargs.pop("batch_size", 1)
        self.log_freq = kwargs.pop("log_freq", 100)
        
        # Call parent init with remaining kwargs
        super().__init__(**kwargs)


class HybridMemoryAttention(nn.Module):
    """Attention layer with both internal and external memory."""
    
    def __init__(
        self,
        config: HybridTransformerConfig,
        self_attn: nn.Module,
        layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.self_attn = self_attn
        self.layer_idx = layer_idx
        
        # Internal memory module (LM2-style)
        if config.use_internal_memory:
            head_size = config.memory_slots // config.num_mem_heads
            self.internal_memory = InternalMemoryModule(
                mem_slots=config.memory_slots,
                head_size=head_size,
                hidden_dim=config.hidden_size,
                num_heads=config.num_mem_heads,
            )
        else:
            self.internal_memory = None
        
        # External memory will be managed at model level
        self.use_external_memory = config.use_external_memory
        
        # Joint attention for fusing retrieved memories
        if self.use_external_memory:
            self.num_heads = config.num_attention_heads
            self.head_dim = config.hidden_size // self.num_heads
            
            # Projection for external memory fusion
            self.external_mem_proj = nn.Linear(
                config.hidden_size,
                config.hidden_size,
                bias=False
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        internal_memory: Optional[torch.Tensor],
        external_kv: Optional[Dict[str, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        position_embeddings: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with hybrid memory.
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_dim]
            internal_memory: Internal memory state [batch, mem_slots, mem_slots]
            external_kv: Retrieved external memory keys/values
            attention_mask: Attention mask
            position_ids: Position IDs
            position_embeddings: Rotary position embeddings
        
        Returns:
            output: Attention output
            updated_internal_memory: Updated internal memory state
        """
        # Standard self-attention
        # LlamaAttention returns (attn_output, attn_weights, past_key_values) when use_cache=True
        # or just (attn_output, None) when use_cache=False
        attn_result = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,  # Updated from past_key_value to past_key_values
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=position_embeddings,
        )
        # Handle both return formats
        if isinstance(attn_result, tuple):
            attn_output = attn_result[0]
        else:
            attn_output = attn_result
        
        # Apply internal memory if enabled
        updated_internal_memory = None
        if self.internal_memory is not None and internal_memory is not None:
            gated_memory, updated_internal_memory = self.internal_memory(
                attn_output,
                internal_memory,
                attention_mask
            )
            attn_output = attn_output + gated_memory
        
        # Fuse external memory if available
        if self.use_external_memory and external_kv is not None:
            attn_output = self._fuse_external_memory(
                attn_output,
                external_kv,
                attention_mask
            )
        
        return attn_output, updated_internal_memory
    
    def _fuse_external_memory(
        self,
        hidden_states: torch.Tensor,
        external_kv: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Fuse retrieved external memories using joint attention."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Reshape for multi-head attention
        query = hidden_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        # External keys and values: [batch*num_heads, seq_len, k, head_dim]
        ext_keys = external_kv['k']
        ext_values = external_kv['v']
        
        # Reshape external memories
        ext_keys = ext_keys.view(
            batch_size, self.num_heads, seq_len, -1, self.head_dim
        )  # [batch, num_heads, seq_len, k, head_dim]
        ext_values = ext_values.view(
            batch_size, self.num_heads, seq_len, -1, self.head_dim
        )
        
        # Compute attention scores with external memories
        scores = torch.einsum(
            'bhqd,bhqkd->bhqk',
            query,
            ext_keys
        ) / (self.head_dim ** 0.5)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum of external values
        external_context = torch.einsum(
            'bhqk,bhqkd->bhqd',
            attn_weights,
            ext_values
        )
        
        # Reshape and project
        external_context = external_context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim
        )
        external_context = self.external_mem_proj(external_context)
        
        # Residual connection
        return hidden_states + external_context


class HybridTransformerModel(PreTrainedModel):
    """
    Hybrid Transformer model with dual memory systems.
    Works with any transformer architecture (Llama, Qwen, GPT, etc.)
    """
    
    config_class = HybridTransformerConfig
    
    def __init__(self, config: HybridTransformerConfig, tokenizer: AutoTokenizer):
        super().__init__(config)
        self.config = config
        self.tokenizer = tokenizer
        
        # Create base config without memory parameters
        base_config = AutoConfig.for_model(
            model_type=config.model_type if hasattr(config, 'model_type') and config.model_type != 'hybrid_transformer' else 'llama',
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=getattr(config, 'rms_norm_eps', 1e-6),
            rope_theta=getattr(config, 'rope_theta', 10000.0),
            attention_dropout=getattr(config, 'attention_dropout', 0.0),
        )
        
        # Load base model from config
        base_model = AutoModelForCausalLM.from_config(base_config)
        self.model = base_model.model  # Get the base transformer
        self.lm_head = base_model.lm_head
        
        # Initialize internal memory
        if config.use_internal_memory:
            self.internal_memory = torch.stack([
                torch.eye(config.memory_slots, requires_grad=False)
                for _ in range(config.batch_size)
            ])
            self.register_buffer("internal_memory_bank", self.internal_memory)
        else:
            self.internal_memory = None
        
        # Initialize external memory bank
        if config.use_external_memory:
            self.external_memory = ExternalMemoryBank(config)
        else:
            self.external_memory = None
        
        # Wrap decoder layers with hybrid memory (if they exist)
        if hasattr(self.model, 'layers'):
            self._wrap_decoder_layers()
    
    def _wrap_decoder_layers(self):
        """Wrap existing decoder layers with hybrid memory attention."""
        original_layers = self.model.layers
        
        class HybridDecoderLayer(nn.Module):
            def __init__(self, original_layer, config, layer_idx):
                super().__init__()
                self.original_layer = original_layer
                self.hybrid_attn = HybridMemoryAttention(
                    config,
                    original_layer.self_attn,
                    layer_idx
                )
                self.input_layernorm = original_layer.input_layernorm
                self.post_attention_layernorm = original_layer.post_attention_layernorm
                self.mlp = original_layer.mlp
            
            def forward(
                self,
                hidden_states,
                attention_mask=None,
                position_ids=None,
                internal_memory=None,
                external_kv=None,
                position_embeddings=None,
            ):
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
                
                # Hybrid memory attention
                attn_output, updated_internal_memory = self.hybrid_attn(
                    hidden_states=hidden_states,
                    internal_memory=internal_memory,
                    external_kv=external_kv,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                
                hidden_states = residual + attn_output
                
                # Feed-forward network
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states = self.mlp(hidden_states)
                hidden_states = residual + hidden_states
                
                return hidden_states, updated_internal_memory
        
        # Replace layers
        self.model.layers = nn.ModuleList([
            HybridDecoderLayer(layer, self.config, idx)
            for idx, layer in enumerate(original_layers)
        ])
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_external_memory: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Forward pass with hybrid memory.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            targets: Target token IDs for loss calculation
            attention_mask: Attention mask
            position_ids: Position IDs
            use_external_memory: Whether to use external memory retrieval
        
        Returns:
            logits: Output logits
            loss: Cross-entropy loss (if targets provided)
            memory_info: Dictionary with memory states
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.size()
        
        # Ensure internal memory is on correct device
        if self.internal_memory is not None:
            if self.internal_memory.device != device:
                self.internal_memory = self.internal_memory.to(device)
            internal_memory = self.internal_memory.detach()
        else:
            internal_memory = None
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=device
            ).unsqueeze(0).repeat(batch_size, 1)
        
        # Get embeddings
        inputs_embeds = self.model.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        
        # Create causal attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Create 4D causal mask
        causal_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_len), inputs_embeds, 0
        )
        
        # Get position embeddings
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        
        # Process through decoder layers
        for layer_idx, decoder_layer in enumerate(self.model.layers):
            # Retrieve from external memory if enabled
            external_kv = None
            if (use_external_memory and 
                self.external_memory is not None and
                self.external_memory.is_ready()):
                
                # Get queries for retrieval (use hidden states)
                queries = hidden_states.transpose(0, 1)  # [seq_len, batch, hidden]
                external_kv = self.external_memory.retrieve(queries)
            
            # Forward through layer
            hidden_states, internal_memory = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                internal_memory=internal_memory,
                external_kv=external_kv,
                position_embeddings=position_embeddings,
            )
        
        # Final layer norm and output projection
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states).float()
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.tokenizer.pad_token_id,
            )
        
        # Update internal memory
        if self.internal_memory is not None:
            self.internal_memory = internal_memory
            self.internal_memory_bank = internal_memory.clone()
        
        # Prepare memory info
        memory_info = {
            'internal_memory': internal_memory,
            'external_memory_size': (
                self.external_memory.get_size()
                if self.external_memory is not None else 0
            ),
        }
        
        return logits, loss, memory_info
    
    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        """Create causal attention mask compatible with different transformers versions."""
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = self._expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None 
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    
    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, 
        past_key_values_length: int = 0
    ):
        """Make causal mask for autoregressive generation."""
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([
                torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), 
                mask
            ], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    
    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int = None):
        """Expand attention mask from 2D to 4D."""
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(dtype).min
        )
    
    def configure_optimizers(self, train_config):
        """Configure optimizer with proper parameter groups."""
        import re
        
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (
            torch.nn.LayerNorm,
            torch.nn.Embedding,
        )
        
        # Add RMSNorm if it exists
        try:
            from transformers.models.llama.modeling_llama import LlamaRMSNorm
            blacklist_weight_modules = blacklist_weight_modules + (LlamaRMSNorm,)
        except:
            pass
        
        # Pattern for internal memory parameters
        pattern = re.compile(
            r"^model\.layers\.\d+\.hybrid_attn\.internal_memory\."
            r"input_gate_projector\.w$"
        )
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                
                # Skip lm_head parameters from weight decay categorization
                # They will be added to decay group by default
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pattern.match(fpn):
                    no_decay.add(fpn)
        
        # Validate parameter separation
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        
        # Add any remaining parameters (like lm_head) to decay group
        remaining_params = param_dict.keys() - union_params
        if remaining_params:
            decay.update(remaining_params)
            union_params = decay | no_decay
        
        assert len(inter_params) == 0, (
            f"Parameters in both decay/no_decay: {inter_params}"
        )
        assert len(param_dict.keys() - union_params) == 0, (
            f"Parameters not in either set: {param_dict.keys() - union_params}"
        )
        
        # Create optimizer groups
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=train_config.learning_rate,
            betas=train_config.betas
        )
        return optimizer
