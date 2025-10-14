"""
Internal Memory Module (adapted from LM2).
Provides learnable memory slots for short-term context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class InternalMemoryModule(nn.Module):
    """
    Internal memory module with gated updates.
    Based on LM2's memory architecture.
    """
    
    def __init__(
        self,
        mem_slots: int,
        head_size: int,
        hidden_dim: int,
        num_heads: int,
    ):
        """
        Initialize internal memory module.
        
        Args:
            mem_slots: Number of memory slots
            head_size: Size of each memory head
            hidden_dim: Hidden dimension of the model
            num_heads: Number of memory heads
        """
        super().__init__()
        self.mem_slots = mem_slots
        self.head_size = head_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Input gate projector
        self.input_gate_projector = nn.Linear(hidden_dim, mem_slots, bias=False)
        
        # Memory read projector
        self.memory_read_projector = nn.Linear(mem_slots, hidden_dim, bias=False)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Memory update projector
        self.memory_update_projector = nn.Linear(hidden_dim, mem_slots, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through memory module.
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_dim]
            memory: Current memory state [batch, mem_slots, mem_slots]
            attention_mask: Optional attention mask
        
        Returns:
            gated_output: Gated memory output
            updated_memory: Updated memory state
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Project hidden states to memory space
        # [batch, seq_len, mem_slots]
        input_projection = self.input_gate_projector(hidden_states)
        
        # Read from memory
        # [batch, seq_len, mem_slots] @ [batch, mem_slots, mem_slots]
        # -> [batch, seq_len, mem_slots]
        memory_read = torch.bmm(input_projection, memory)
        
        # Project memory read back to hidden dimension
        # [batch, seq_len, hidden_dim]
        memory_output = self.memory_read_projector(memory_read)
        
        # Gating mechanism
        gate_input = torch.cat([hidden_states, memory_output], dim=-1)
        gate_values = self.gate(gate_input)
        
        # Gated output
        gated_output = gate_values * memory_output
        
        # Update memory
        # Project hidden states for memory update
        update_projection = self.memory_update_projector(hidden_states)
        
        # Compute memory update using outer product
        # Average over sequence length
        # [batch, mem_slots, seq_len] @ [batch, seq_len, mem_slots]
        # -> [batch, mem_slots, mem_slots]
        memory_update = torch.bmm(
            update_projection.transpose(1, 2),
            input_projection
        ) / seq_len
        
        # Apply update with decay
        decay_factor = 0.9
        updated_memory = decay_factor * memory + (1 - decay_factor) * memory_update
        
        # Normalize memory to maintain stability
        updated_memory = F.normalize(updated_memory, p=2, dim=-1)
        
        return gated_output, updated_memory
