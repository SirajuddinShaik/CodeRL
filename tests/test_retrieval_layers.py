"""Test selective layer retrieval configuration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mem_aug.components.memory.hybrid_model import HybridTransformerConfig


def test_retrieval_layers():
    """Test retrieval layer calculation with different configurations."""
    
    print("=" * 80)
    print("TESTING SELECTIVE LAYER RETRIEVAL")
    print("=" * 80)
    
    test_cases = [
        {
            "name": "Default (L//6)",
            "num_layers": 36,
            "num_retrieval_layers": None,
            "expected": [0, 6, 12, 18, 24, 30]
        },
        {
            "name": "Custom R=6",
            "num_layers": 36,
            "num_retrieval_layers": 6,
            "expected": [0, 6, 12, 18, 24, 30]
        },
        {
            "name": "Custom R=12",
            "num_layers": 36,
            "num_retrieval_layers": 12,
            "expected": [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]
        },
        {
            "name": "All layers (R=36)",
            "num_layers": 36,
            "num_retrieval_layers": 36,
            "expected": list(range(36))
        },
        {
            "name": "Single layer (R=1)",
            "num_layers": 36,
            "num_retrieval_layers": 1,
            "expected": [0]
        },
        {
            "name": "Small model (L=12, default)",
            "num_layers": 12,
            "num_retrieval_layers": None,
            "expected": [0, 6]
        },
        {
            "name": "Tiny model (L=6, default)",
            "num_layers": 6,
            "num_retrieval_layers": None,
            "expected": [0]
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}: {test['name']}")
        print(f"{'=' * 80}")
        
        # Create config
        config = HybridTransformerConfig(
            vocab_size=32000,
            hidden_size=2048,
            num_hidden_layers=test['num_layers'],
            num_attention_heads=16,
            intermediate_size=8192,
            max_position_embeddings=4096,
            num_retrieval_layers=test['num_retrieval_layers']
        )
        
        # Get retrieval layers
        retrieval_layers = config.get_retrieval_layers()
        
        # Display results
        print(f"\nConfiguration:")
        print(f"  Total layers (L): {test['num_layers']}")
        print(f"  Retrieval layers param (R): {test['num_retrieval_layers']}")
        print(f"  Actual R: {len(retrieval_layers)}")
        
        print(f"\nRetrieving at layers: {retrieval_layers}")
        print(f"Expected layers:      {test['expected']}")
        
        # Verify
        if retrieval_layers == test['expected']:
            print(f"\n✅ TEST {i} PASSED")
        else:
            print(f"\n❌ TEST {i} FAILED")
            print(f"   Got:      {retrieval_layers}")
            print(f"   Expected: {test['expected']}")
            return False
    
    print(f"\n{'=' * 80}")
    print("ALL TESTS PASSED! ✅")
    print(f"{'=' * 80}")
    
    # Summary
    print("\nSummary:")
    print("  - Default behavior: R = L // 6")
    print("  - Custom R can be specified via num_retrieval_layers parameter")
    print("  - Layers are evenly distributed across model depth")
    print("  - Formula: layer_idx = int(k * L / R) for k in range(R)")
    
    return True


if __name__ == "__main__":
    success = test_retrieval_layers()
    sys.exit(0 if success else 1)
