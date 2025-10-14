"""
Test script to load the hybrid model and perform inference.
Shows output from randomly initialized model.
"""

import torch
from transformers import AutoTokenizer, AutoConfig
from mem_aug.components.memory import HybridTransformerModel, HybridTransformerConfig

def test_model_inference():
    """Test model loading and inference."""
    
    print("=" * 80)
    print("HYBRID LLM MODEL INFERENCE TEST")
    print("=" * 80)
    
    # Configuration
    model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n1. Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   ✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    print(f"\n2. Loading base model config from: {model_name}")
    base_config = AutoConfig.from_pretrained(model_name)
    print(f"   ✓ Base config loaded:")
    print(f"     - Hidden size: {base_config.hidden_size}")
    print(f"     - Num layers: {base_config.num_hidden_layers}")
    print(f"     - Num attention heads: {base_config.num_attention_heads}")
    print(f"     - Vocab size: {base_config.vocab_size}")
    print(f"     - Max position embeddings: {base_config.max_position_embeddings}")
    
    print(f"\n3. Creating hybrid config with memory modules")
    hybrid_config = HybridTransformerConfig(
        # Base model parameters (from loaded config)
        vocab_size=base_config.vocab_size,
        hidden_size=base_config.hidden_size,
        num_hidden_layers=base_config.num_hidden_layers,
        num_attention_heads=base_config.num_attention_heads,
        intermediate_size=base_config.intermediate_size,
        max_position_embeddings=base_config.max_position_embeddings,
        rms_norm_eps=getattr(base_config, 'rms_norm_eps', 1e-6),
        rope_theta=getattr(base_config, 'rope_theta', 10000.0),
        
        # Memory parameters
        use_internal_memory=True,
        memory_slots=16,
        num_mem_heads=4,
        use_external_memory=True,  # Disable for initial test
        batch_size=1,
        log_freq=100,
    )
    print(f"   ✓ Hybrid config created")
    print(f"     - Internal memory: {hybrid_config.use_internal_memory}")
    print(f"     - Memory slots: {hybrid_config.memory_slots}")
    print(f"     - Memory heads: {hybrid_config.num_mem_heads}")
    
    print(f"\n4. Initializing hybrid model (randomly initialized)")
    model = HybridTransformerModel(hybrid_config, tokenizer)
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Model initialized on {device}")
    print(f"     - Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"     - Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    print(f"\n5. Preparing test input")
    test_prompts = [
        "Write a Python function to calculate fibonacci numbers:",
        "Explain what is machine learning:",
        "Translate to French: Hello, how are you?",
    ]
    
    print(f"\n{'=' * 80}")
    print("INFERENCE RESULTS (Random Initialization)")
    print(f"{'=' * 80}\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}/{len(test_prompts)}")
        print(f"Prompt: {prompt}")
        print("-" * 80)
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs['input_ids']
        
        print(f"Input tokens: {input_ids.shape[1]}")
        
        # Generate (greedy decoding for simplicity)
        with torch.no_grad():
            max_new_tokens = 40
            generated_ids = input_ids.clone()
            
            for step in range(max_new_tokens):
                # Forward pass
                logits, loss, memory_info = model(
                    input_ids=generated_ids,
                    use_external_memory=False,
                )
                
                # Get next token (greedy)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Stop if EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Decode output
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        output_only = generated_text[len(prompt):].strip()
        
        print(f"\nGenerated output ({generated_ids.shape[1] - input_ids.shape[1]} tokens):")
        print(f"'{output_only}'")
        
        # Show memory info
        if memory_info.get('internal_memory') is not None:
            mem_shape = memory_info['internal_memory'].shape
            print(f"\nInternal memory state: {mem_shape}")
        
        print(f"\n{'=' * 80}\n")
    
    print("\n✓ Inference test completed!")
    print("\nNote: Output is random/gibberish because model is randomly initialized.")
    print("After training on input/output pairs, the model will generate coherent responses.")
    print("=" * 80)


if __name__ == "__main__":
    test_model_inference()
