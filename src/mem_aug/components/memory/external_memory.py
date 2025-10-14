"""
External Memory Bank (adapted from LongMem).
Provides FAISS-based retrieval for long-term context.
"""

import torch
import numpy as np
from typing import Dict, Optional

try:
    import faiss
    import faiss.contrib.torch_utils
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. External memory will be disabled.")


class ExternalMemoryBank:
    """
    External memory bank using FAISS for efficient retrieval.
    Based on LongMem's dynamic memory architecture.
    """
    
    def __init__(self, config):
        """
        Initialize external memory bank.
        
        Args:
            config: Model configuration with memory parameters
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is required for external memory. "
                "Install with: pip install faiss-gpu or faiss-cpu"
            )
        
        self.dimension = config.hidden_size
        self.use_gpu_to_search = config.use_gpu_to_search
        self.k = config.retrieval_k
        self.memory_size = config.external_memory_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dimension // self.num_heads
        self.chunk_size = config.chunk_size
        
        # Initialize FAISS indices (one per attention head)
        self._initialize_indices()
        
        # Initialize key-value storage
        self._initialize_storage()
        
        self.dstore_idx = 0
        self.time_for_retrieve = 0.0
        self.retrieve_count = 0
    
    def _initialize_indices(self):
        """Initialize FAISS indices for each attention head."""
        self.index_list = []
        
        if self.use_gpu_to_search and torch.cuda.is_available():
            print(f'Initializing GPU FAISS indices on device {torch.cuda.current_device()}')
            self.res = faiss.StandardGpuResources()
            
            for i in range(self.num_heads):
                cpu_index = faiss.IndexFlatIP(self.head_dim)
                gpu_index = faiss.index_cpu_to_gpu(
                    self.res,
                    torch.cuda.current_device(),
                    cpu_index
                )
                self.index_list.append(gpu_index)
        else:
            print('Initializing CPU FAISS indices')
            self.index_list = [
                faiss.IndexFlatIP(self.head_dim)
                for _ in range(self.num_heads)
            ]
    
    def _initialize_storage(self):
        """Initialize key-value storage tensors."""
        storage_size = self.memory_size // self.chunk_size
        
        if self.use_gpu_to_search and torch.cuda.is_available():
            device = torch.cuda.current_device()
            dtype = torch.float16
        else:
            device = 'cpu'
            dtype = torch.float32
        
        self.keys = [
            torch.zeros(
                storage_size,
                self.chunk_size,
                self.head_dim,
                dtype=dtype,
                device=device
            )
            for _ in range(self.num_heads)
        ]
        
        self.vals = [
            torch.zeros(
                storage_size,
                self.chunk_size,
                self.head_dim,
                dtype=dtype,
                device=device
            )
            for _ in range(self.num_heads)
        ]
    
    def reset(self):
        """Reset the memory bank."""
        self.dstore_idx = 0
        
        # Reset FAISS indices
        for index in self.index_list:
            index.reset()
        
        # Reinitialize storage
        self._initialize_storage()
        
        self.time_for_retrieve = 0.0
        self.retrieve_count = 0
    
    def add_index(
        self,
        qkv_val: Dict[str, torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None
    ):
        """
        Add key-value pairs to the memory bank.
        
        Args:
            qkv_val: Dictionary with 'k' and 'v' tensors
                    Shape: [batch, seq_len, num_heads, head_dim]
            padding_mask: Optional padding mask
        """
        keys = qkv_val['k']
        vals = qkv_val['v']
        
        batch_size, seq_len, num_heads, head_dim = keys.shape
        
        # Check if memory is full and needs cleanup
        chunks_to_add = (batch_size * seq_len) // self.chunk_size
        max_chunks = self.memory_size // self.chunk_size
        
        if self.dstore_idx + chunks_to_add >= max_chunks:
            # Remove oldest entries
            update_size = min(chunks_to_add * 2, max_chunks // 2)
            self._remove_oldest(update_size)
        
        # Reshape keys and values
        keys = keys.reshape(batch_size * seq_len, num_heads, head_dim)
        vals = vals.reshape(batch_size * seq_len, num_heads, head_dim)
        
        # Keep only complete chunks
        keep_dim = (batch_size * seq_len) // self.chunk_size * self.chunk_size
        
        if keep_dim == 0:
            return  # Not enough data for even one chunk
        
        keys = keys[:keep_dim]
        vals = vals[:keep_dim]
        
        # Reshape into chunks
        num_chunks = keep_dim // self.chunk_size
        keys_chunked = keys.reshape(
            num_chunks, self.chunk_size, num_heads, head_dim
        )
        vals_chunked = vals.reshape(
            num_chunks, self.chunk_size, num_heads, head_dim
        )
        
        # Add to each head's index
        for i, index in enumerate(self.index_list):
            # Compute chunk representatives (mean over chunk)
            chunk_keys = keys_chunked[:, :, i, :].mean(dim=1)
            
            # Add to FAISS index
            if self.use_gpu_to_search:
                index.add(chunk_keys.float().contiguous())
            else:
                index.add(chunk_keys.cpu().float().numpy())
            
            # Store full keys and values
            end_idx = self.dstore_idx + num_chunks
            self.keys[i][self.dstore_idx:end_idx] = keys_chunked[:, :, i, :]
            self.vals[i][self.dstore_idx:end_idx] = vals_chunked[:, :, i, :]
        
        self.dstore_idx += num_chunks
    
    def _remove_oldest(self, num_chunks: int):
        """Remove oldest entries from memory."""
        if self.use_gpu_to_search:
            for i, index in enumerate(self.index_list):
                # Convert to CPU, remove, convert back
                cpu_index = faiss.index_gpu_to_cpu(index)
                cpu_index.remove_ids(np.arange(num_chunks))
                gpu_index = faiss.index_cpu_to_gpu(
                    self.res,
                    torch.cuda.current_device(),
                    cpu_index
                )
                self.index_list[i] = gpu_index
        else:
            for index in self.index_list:
                index.remove_ids(np.arange(num_chunks))
        
        # Shift storage
        for i in range(self.num_heads):
            self.keys[i] = torch.cat([
                self.keys[i][num_chunks:],
                torch.zeros_like(self.keys[i][:num_chunks])
            ])
            self.vals[i] = torch.cat([
                self.vals[i][num_chunks:],
                torch.zeros_like(self.vals[i][:num_chunks])
            ])
        
        self.dstore_idx = max(0, self.dstore_idx - num_chunks)
    
    def retrieve(
        self,
        queries: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieve nearest neighbors from memory.
        
        Args:
            queries: Query tensor [seq_len, batch, hidden_dim]
        
        Returns:
            Dictionary with retrieved keys and values
        """
        seq_len, batch_size, hidden_dim = queries.shape
        
        # Reshape queries for multi-head
        queries = queries.reshape(
            seq_len * batch_size, self.num_heads, self.head_dim
        ).float()
        
        # Search in each head's index
        k_per_chunk = self.k // self.chunk_size
        indices_list = []
        
        for i in range(self.num_heads):
            query_head = queries[:, i, :].contiguous()
            
            if self.use_gpu_to_search:
                _, indices = self.index_list[i].search(query_head, k_per_chunk)
            else:
                _, indices = self.index_list[i].search(
                    query_head.cpu().numpy(),
                    k_per_chunk
                )
                indices = torch.from_numpy(indices).to(queries.device)
            
            indices_list.append(indices)
        
        # Retrieve keys and values
        retrieved_keys = []
        retrieved_vals = []
        
        for i in range(self.num_heads):
            indices = indices_list[i]  # [seq_len*batch, k_per_chunk]
            
            # Gather keys and values
            head_keys = self.keys[i][indices]  # [seq_len*batch, k_per_chunk, chunk_size, head_dim]
            head_vals = self.vals[i][indices]
            
            # Reshape to [seq_len*batch, k, head_dim]
            head_keys = head_keys.reshape(seq_len * batch_size, self.k, self.head_dim)
            head_vals = head_vals.reshape(seq_len * batch_size, self.k, self.head_dim)
            
            retrieved_keys.append(head_keys)
            retrieved_vals.append(head_vals)
        
        # Stack across heads
        # [num_heads, seq_len*batch, k, head_dim] -> [seq_len*batch*num_heads, k, head_dim]
        keys_stacked = torch.stack(retrieved_keys, dim=0)
        vals_stacked = torch.stack(retrieved_vals, dim=0)
        
        keys_stacked = keys_stacked.transpose(0, 1).reshape(
            seq_len * batch_size * self.num_heads, self.k, self.head_dim
        )
        vals_stacked = vals_stacked.transpose(0, 1).reshape(
            seq_len * batch_size * self.num_heads, self.k, self.head_dim
        )
        
        # Reshape to [batch*num_heads, seq_len, k, head_dim]
        keys_final = keys_stacked.view(
            seq_len, batch_size * self.num_heads, self.k, self.head_dim
        ).transpose(0, 1)
        
        vals_final = vals_stacked.view(
            seq_len, batch_size * self.num_heads, self.k, self.head_dim
        ).transpose(0, 1)
        
        return {
            'k': keys_final,
            'v': vals_final,
            'indices': indices_list
        }
    
    def is_ready(self) -> bool:
        """Check if memory bank has enough entries for retrieval."""
        return self.dstore_idx > 0
    
    def get_size(self) -> int:
        """Get current number of stored chunks."""
        return self.dstore_idx
    
    def get_stats(self) -> Dict[str, float]:
        """Get retrieval statistics."""
        avg_time = (
            self.time_for_retrieve / self.retrieve_count
            if self.retrieve_count > 0 else 0.0
        )
        
        return {
            'total_retrievals': self.retrieve_count,
            'avg_retrieval_time': avg_time,
            'memory_usage': self.dstore_idx / (self.memory_size // self.chunk_size),
        }
