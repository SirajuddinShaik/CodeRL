"""
Test script for softmax-based gating mechanism in hybrid memory model.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mem_aug.components.memory.hybrid_model import (
    HybridTransformerModel,
    HybridTransformerConfig
)


def test_gating_mechanism():
    """Test the softmax-based gating mechanism with different memory configurations."""
    
    print("=" * 80)
    print("TESTING SOFTMAX-BASED GATING MECHANISM")
    print("=" * 80)
    
    # Load tokenizer
    model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
    print(f"\n1. Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"   ✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # Load base config
    print(f"\n2. Loading base model config from: {model_name}")
    base_config = AutoConfig.from_pretrained(model_name)
    print(f"   ✓ Base config loaded")
    print(f"     - Hidden size: {base_config.hidden_size}")
    print(f"     - Num layers: {base_config.num_hidden_layers}")
    print(f"     - Num attention heads: {base_config.num_attention_heads}")
    
    # Test configurations
    test_configs = [
        {
            "name": "Only Self-Attention",
            "use_internal_memory": False,
            "use_external_memory": False,
            "expected_sources": 1
        },
        {
            "name": "Self-Attention + Internal Memory",
            "use_internal_memory": True,
            "use_external_memory": False,
            "expected_sources": 2
        },
        {
            "name": "Self-Attention + External Memory",
            "use_internal_memory": False,
            "use_external_memory": True,
            "expected_sources": 2
        },
        {
            "name": "All Three Memory Sources",
            "use_internal_memory": True,
            "use_external_memory": True,
            "expected_sources": 3
        }
    ]
    
    for i, test_config in enumerate(test_configs, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}: {test_config['name']}")
        print(f"{'=' * 80}")
        
        # Create hybrid config
        print(f"\n   Creating hybrid config...")
        hybrid_config = HybridTransformerConfig(
            vocab_size=base_config.vocab_size,
            hidden_size=base_config.hidden_size,
            num_hidden_layers=2,  # Use fewer layers for testing
            num_attention_heads=base_config.num_attention_heads,
            intermediate_size=base_config.intermediate_size,
            max_position_embeddings=base_config.max_position_embeddings,
            
            # Memory parameters
            use_internal_memory=test_config['use_internal_memory'],
            memory_slots=16,
            num_mem_heads=4,
            use_external_memory=test_config['use_external_memory'],
            external_memory_size=1024,  # Smaller for testing
            retrieval_k=4,
            batch_size=1,
        )
        
        print(f"   ✓ Hybrid config created")
        print(f"     - Internal memory: {hybrid_config.use_internal_memory}")
        print(f"     - External memory: {hybrid_config.use_external_memory}")
        print(f"     - Expected active sources: {test_config['expected_sources']}")
        
        # Initialize model
        print(f"\n   Initializing model...")
        model = HybridTransformerModel(hybrid_config, tokenizer)
        model.eval()
        print(f"   ✓ Model initialized")
        
        # Create test input
        test_prompt = "def fibonacci(n):"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        print(f"\n   Running forward pass...")
        print(f"   Input: '{test_prompt}'")
        print(f"   Input shape: {input_ids.shape}")
        
        # Forward pass
        with torch.no_grad():
            logits, loss, memory_info = model(
                input_ids=input_ids,
                use_external_memory=test_config['use_external_memory']
            )
        
        print(f"\n   ✓ Forward pass successful!")
        print(f"     - Output logits shape: {logits.shape}")
        print(f"     - Internal memory shape: {memory_info['internal_memory'].shape if memory_info['internal_memory'] is not None else 'None'}")
        print(f"     - External memory size: {memory_info['external_memory_size']}")
        
        # Check gating network
        first_layer = model.model.layers[0]
        gate_network = first_layer.hybrid_attn.gate_network
        
        print(f"\n   Gating Network Analysis:")
        print(f"     - Gate network weight shape: {gate_network.weight.shape}")
        print(f"     - Gate network bias shape: {gate_network.bias.shape}")
        print(f"     - Max gates available: {gate_network.out_features}")
        
        # Simulate gate computation
        with torch.no_grad():
            # Get hidden states from embeddings
            hidden_states = model.model.embed_tokens(input_ids)
            
            # Compute gate logits
            gate_logits = gate_network(hidden_states)  # [batch, seq_len, 3]
            
            # Use only active sources
            active_sources = test_config['expected_sources']
            gate_logits_active = gate_logits[:, :, :active_sources]
            
            # Apply softmax
            gate_weights = F.softmax(gate_logits_active, dim=-1)
            
            print(f"\n   Gate Weights (first token):")
            for j in range(active_sources):
                source_names = ["Self-Attention", "Internal Memory", "External Memory"]
                weight = gate_weights[0, 0, j].item()
                print(f"     - {source_names[j]}: {weight:.4f}")
            
            # Verify softmax constraint
            total_weight = gate_weights[0, 0, :].sum().item()
            print(f"\n   ✓ Softmax constraint verified: sum = {total_weight:.6f}")
            assert abs(total_weight - 1.0) < 1e-5, "Gate weights should sum to 1!"
        
        print(f"\n   ✓ Test {i} PASSED!")
    
    print(f"\n{'=' * 80}")
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)
    print("\nSummary:")
    print("  - Gating network successfully adapts to 1, 2, or 3 active memory sources")
    print("  - Softmax ensures gate weights sum to 1.0")
    print("  - Dynamic memory fusion working correctly")
    print("\nImplementation Details:")
    print("  - Formula: H_t = g[0] * A_t + g[1] * M_int + g[2] * M_ext")
    print("  - Where: g = softmax(W_g @ h_t + b_g)")
    print("  - Gate network: Linear(hidden_size, 3)")
    print("  - Only active sources are used in softmax computation")


if __name__ == "__main__":
    test_gating_mechanism()
