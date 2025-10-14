# Hybrid LLM Trainer

A unified training framework combining LM2's memory-augmented architecture with LongMem's external memory retrieval for efficient input/output training on large language models.

## Overview

This hybrid module integrates:
- **LM2**: Memory-augmented Llama architecture with learnable memory slots
- **LongMem**: External memory bank with FAISS-based retrieval for long-context modeling
- **Unified Training**: Streamlined training pipeline for input/output pairs

## Key Features

1. **Dual Memory System**:
   - Internal memory slots (from LM2) for short-term context
   - External memory bank (from LongMem) for long-term retrieval

2. **Flexible Architecture**:
   - Support for various base models (Llama, GPT-2, etc.)
   - Configurable memory parameters
   - Hybrid attention mechanisms

3. **Efficient Training**:
   - Distributed training support (DDP)
   - Mixed precision training
   - Checkpoint management

## Installation

```bash
# Create conda environment
conda env create -n hybrid_llm -f environment.yaml
conda activate hybrid_llm

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

```bash
# Prepare input/output training data
python data_proc/prepare_io_data.py \
  --input_file data/train_inputs.jsonl \
  --output_file data/train_outputs.jsonl \
  --save_dir datasets/io_pairs
```

### 2. Train the Model

```bash
# Basic training
python train.py \
  model=llama_hybrid \
  train.batch_size=4 \
  train.learning_rate=1e-4

# Or use the training script
bash scripts/train_hybrid.sh
```

### 3. Evaluate

```bash
python eval.py \
  --checkpoint checkpoints/best_model.pt \
  --test_data datasets/io_pairs/test
```

## Architecture

### Hybrid Memory Module

```
Input Sequence
     ↓
Embedding Layer
     ↓
┌─────────────────────────────────┐
│  Transformer Layers             │
│  ┌──────────────────────────┐  │
│  │ Self-Attention           │  │
│  │ ↓                        │  │
│  │ Internal Memory Module   │  │ ← LM2 Memory Slots
│  │ ↓                        │  │
│  │ External Memory Retrieval│  │ ← LongMem FAISS Index
│  │ ↓                        │  │
│  │ Joint Attention Fusion   │  │
│  │ ↓                        │  │
│  │ Feed-Forward Network     │  │
│  └──────────────────────────┘  │
└─────────────────────────────────┘
     ↓
Output Logits
```

## Configuration

Key configuration parameters in `configs/train.yaml`:

```yaml
model:
  model_type: llama_hybrid
  use_internal_memory: true
  use_external_memory: true
  memory_slots: 16
  num_mem_heads: 4
  external_memory_size: 1048576
  retrieval_k: 8

train:
  batch_size: 4
  learning_rate: 1e-4
  max_iters: 100000
  dtype: bfloat16
```

## Project Structure

```
hybrid_llm_trainer/
├── configs/              # Configuration files
│   ├── train.yaml       # Main training config
│   └── model/           # Model-specific configs
├── data_proc/           # Data processing scripts
│   ├── prepare_io_data.py
│   └── data_utils.py
├── src/
│   ├── models/
│   │   ├── hybrid_llama.py      # Hybrid Llama model
│   │   ├── internal_memory.py   # LM2-style memory
│   │   └── external_memory.py   # LongMem-style retrieval
│   ├── dataloader.py    # Data loading utilities
│   ├── trainer.py       # Training loop
│   └── utils.py         # Helper functions
├── scripts/
│   └── train_hybrid.sh  # Training script
├── train.py             # Main training entry point
├── eval.py              # Evaluation script
├── requirements.txt
└── README.md
```

## Training on Input/Output Pairs

The hybrid trainer is specifically designed for input/output training:

1. **Input Phase**: Model processes input with both memory systems active
2. **Memory Update**: Internal and external memories are updated with input context
3. **Output Phase**: Model generates output leveraging both memory systems
4. **Loss Calculation**: Cross-entropy loss on output tokens only

Example data format:
```json
{
  "input": "Translate to French: Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

## Advanced Usage

### Custom Memory Configuration

```python
from src.models.hybrid_llama import HybridLlamaConfig

config = HybridLlamaConfig(
    use_internal_memory=True,
    use_external_memory=True,
    memory_slots=32,
    num_mem_heads=8,
    external_memory_size=2097152,
    retrieval_k=16,
    chunk_size=4
)
```

### Multi-GPU Training

```bash
torchrun --nproc_per_node=4 train.py \
  model=llama_hybrid \
  train.batch_size=2
```

## Performance Tips

1. **Memory Management**:
   - Adjust `memory_slots` based on GPU memory
   - Use `external_memory_size` for long-context tasks
   - Enable `use_gpu_to_search` for faster retrieval

2. **Training Efficiency**:
   - Use mixed precision (`dtype=bfloat16`)
   - Gradient accumulation for larger effective batch sizes
   - Checkpoint activations for memory-intensive models

3. **Data Processing**:
   - Pre-tokenize datasets for faster loading
   - Use appropriate sequence lengths
   - Balance input/output lengths

## Citation

If you use this hybrid trainer, please cite both original papers:

```bibtex
@article{LM2,
  title={LM2: Large Memory Models},
  author={...},
  journal={arXiv preprint arXiv:2502.06049v1},
  year={2025}
}

@article{LongMem,
  title={Augmenting Language Models with Long-Term Memory},
  author={Wang, Weizhi and Dong, Li and Cheng, Hao and Liu, Xiaodong and Yan, Xifeng and Gao, Jianfeng and Wei, Furu},
  journal={arXiv preprint arXiv:2306.07174},
  year={2023}
}
```

## License

This project combines components from LM2 (CC BY-NC 4.0) and LongMem (MIT License). Please refer to individual licenses for specific terms.

## Contributing

Contributions are welcome! Please submit issues and pull requests.

## Support

For questions and issues, please open a GitHub issue or contact the maintainers.
