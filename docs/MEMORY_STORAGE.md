# 💾 Memory Storage: What Data is Stored Where?

## Overview

This document explains exactly what data is stored in the Internal Memory (LM2) and External Memory (FAISS), how it's stored, and how it's used during training and inference.

---

## 🧠 Internal Memory (LM2) - Short-Term Learnable Memory

### What is Stored:

**Abstract Learned Representations** - NOT raw text or tokens!

The internal memory stores **learned patterns** in the form of continuous vectors that the model learns to use during training.

### Storage Format:

```python
internal_memory: torch.Tensor
Shape: [batch_size, memory_slots, memory_slots]
Example: [1, 16, 16]  # 16 slots, each 16-dimensional

# Initial state (identity matrix)
[[1, 0, 0, ..., 0],
 [0, 1, 0, ..., 0],
 [0, 0, 1, ..., 0],
 ...
 [0, 0, 0, ..., 1]]

# After training (learned patterns)
[[0.82, 0.15, 0.03, ..., 0.01],
 [0.12, 0.76, 0.08, ..., 0.04],
 [0.05, 0.11, 0.81, ..., 0.03],
 ...
 [0.02, 0.04, 0.06, ..., 0.88]]
```

### What Does Each Slot Represent?

Each of the 16 memory slots learns to capture different aspects:

1. **Slot 1-4**: Might learn **syntactic patterns**
   - Subject-verb agreement
   - Tense consistency
   - Sentence structure

2. **Slot 5-8**: Might learn **semantic patterns**
   - Topic continuity
   - Argument structure
   - Logical flow

3. **Slot 9-12**: Might learn **contextual patterns**
   - Discourse markers
   - Reference resolution
   - Conversation state

4. **Slot 13-16**: Might learn **task-specific patterns**
   - Code structure (for coding tasks)
   - Mathematical reasoning (for math tasks)
   - Domain knowledge (for specialized tasks)

**Note**: The model learns what to store in each slot automatically during training!

### How It's Updated:

```python
# During forward pass at each layer:

# 1. Read from memory
memory_read = attention(query=hidden_states, key=memory_slots, value=memory_slots)
# Shape: [batch, seq_len, hidden_dim]

# 2. Compute gate (what to remember/forget)
gate = sigmoid(linear(hidden_states))
# Shape: [batch, seq_len, 1]
# Values: 0.0 (forget) to 1.0 (remember)

# 3. Update memory state
new_memory = gate * new_info + (1 - gate) * old_memory
# Gated update: blend new information with old

# 4. Add to output
output = hidden_states + gate * memory_read
```

### Example During Training:

```
Input: "The cat sat on the mat. It was sleeping."

Layer 1:
  - Slot 1: Learns "cat" is the subject (0.85 activation)
  - Slot 5: Learns "sitting" action (0.72 activation)
  
Layer 10:
  - Slot 3: Learns "It" refers to "cat" (0.91 activation)
  - Slot 7: Learns "sleeping" relates to "cat" (0.88 activation)

Layer 20:
  - Slot 2: Maintains subject continuity (0.79 activation)
  - Slot 9: Tracks narrative flow (0.83 activation)
```

### Persistence:

- **During Training**: Updated after each batch
- **Between Batches**: Persists (carries over)
- **Between Sequences**: Can be reset or maintained
- **Saved in Checkpoint**: Yes, as part of model state

---

## 🗄️ External Memory (FAISS) - Long-Term Retrieval Memory

### What is Stored:

**Key-Value Pairs of Hidden States** - Actual representations from past sequences!

The external memory stores:
1. **Keys**: Hidden state vectors from previous sequences
2. **Values**: Corresponding hidden state vectors
3. **Metadata**: Position information, sequence IDs

### Storage Format:

```python
# FAISS Index Structure
external_memory = {
    'index': faiss.IndexFlatIP,  # Inner product similarity search
    'keys': torch.Tensor,         # [num_stored, hidden_dim]
    'values': torch.Tensor,       # [num_stored, hidden_dim]
    'metadata': {
        'positions': List[int],   # Position in original sequence
        'sequence_ids': List[int], # Which sequence it came from
        'timestamps': List[int],   # When it was added
    }
}

# Example with 1000 stored vectors:
keys.shape:   [1000, 2048]  # 1000 vectors, 2048 dimensions
values.shape: [1000, 2048]  # Corresponding values
```

### What Gets Stored:

Every token's hidden representation from past sequences:

```python
# Example: Processing "Write a Python function to sort a list"

Token 1: "Write"
  - Hidden state: [0.12, -0.34, 0.56, ..., 0.78]  # 2048 dims
  - Stored in FAISS at index 0

Token 2: "a"
  - Hidden state: [0.23, -0.12, 0.45, ..., 0.67]
  - Stored in FAISS at index 1

Token 3: "Python"
  - Hidden state: [0.34, -0.23, 0.67, ..., 0.89]
  - Stored in FAISS at index 2

... and so on for all tokens in all sequences
```

### How It's Used During Retrieval:

```python
# During forward pass:

# 1. Current hidden state (query)
current_hidden = [0.15, -0.28, 0.52, ..., 0.71]  # [2048]

# 2. Search FAISS for top-k similar vectors
similarities, indices = faiss_index.search(current_hidden, k=8)

# Results:
# indices = [342, 156, 789, 23, 567, 891, 234, 445]
# similarities = [0.92, 0.87, 0.85, 0.82, 0.79, 0.76, 0.73, 0.71]

# 3. Retrieve corresponding keys and values
retrieved_keys = keys[indices]    # [8, 2048]
retrieved_values = values[indices] # [8, 2048]

# 4. Attend over retrieved memories
attention_output = attention(
    query=current_hidden,
    key=retrieved_keys,
    value=retrieved_values
)

# 5. Fuse with current representation
enhanced_output = current_hidden + attention_output
```

### Example Retrieval Scenario:

```
Current Input: "How do I sort a list in Python?"

FAISS Search Results (top-8 most similar past tokens):

1. "sort" (from: "Write a Python function to sort a list")
   Similarity: 0.92
   
2. "Python" (from: "Python list comprehension example")
   Similarity: 0.87
   
3. "list" (from: "Create a list of numbers")
   Similarity: 0.85
   
4. "sorted" (from: "Use sorted() function")
   Similarity: 0.82
   
5. "function" (from: "Define a function")
   Similarity: 0.79
   
6. "array" (from: "Sort an array")
   Similarity: 0.76
   
7. "algorithm" (from: "Sorting algorithm")
   Similarity: 0.73
   
8. "method" (from: "List methods")
   Similarity: 0.71

→ Model uses these retrieved memories to generate better response!
```

### Storage Capacity:

```python
# Default configuration
max_tokens = 1,048,576  # 1M tokens
hidden_dim = 2048

# Memory usage
keys_memory = 1,048,576 * 2048 * 4 bytes (float32)
            = 8.6 GB

values_memory = 1,048,576 * 2048 * 4 bytes
              = 8.6 GB

total_memory = ~17.2 GB for full capacity
```

### When Vectors Are Added:

```python
# During training:
for batch in dataloader:
    # Forward pass
    hidden_states = model(batch)
    
    # Add to external memory (every N steps)
    if step % update_freq == 0:
        external_memory.add(
            keys=hidden_states,      # Current hidden states
            values=hidden_states,    # Same as keys
            metadata={
                'sequence_id': batch_id,
                'positions': token_positions,
            }
        )
```

### Persistence:

- **During Training**: Continuously updated
- **Between Batches**: Persists (accumulates)
- **Between Sessions**: Saved to disk as FAISS index file
- **Saved in Checkpoint**: Yes, as separate `.faiss` file

---

## 🔄 How They Work Together

### Complete Flow Example:

```
Input: "Explain recursion in Python"

┌─────────────────────────────────────────────────────────┐
│ Layer 1                                                  │
├─────────────────────────────────────────────────────────┤
│ 1. Self-Attention                                        │
│    - Process "Explain recursion in Python"              │
│    - Output: hidden_states [1, 5, 2048]                 │
│                                                           │
│ 2. Internal Memory (LM2)                                 │
│    - Read from 16 learned slots                         │
│    - Slot 3: "Python" context (0.88)                    │
│    - Slot 7: "Explanation" mode (0.92)                  │
│    - Gate: Keep 85% of memory                           │
│    - Output: hidden_states + gated_memory               │
│                                                           │
│ 3. External Memory (FAISS)                               │
│    - Query: current hidden_states                       │
│    - Retrieved from past:                                │
│      * "recursion" (similarity: 0.94)                   │
│      * "function calls itself" (similarity: 0.89)       │
│      * "base case" (similarity: 0.87)                   │
│      * "Python def" (similarity: 0.85)                  │
│      * "factorial example" (similarity: 0.82)           │
│      * "stack overflow" (similarity: 0.79)              │
│      * "recursive function" (similarity: 0.76)          │
│      * "return statement" (similarity: 0.73)            │
│    - Attend over retrieved memories                     │
│    - Output: enhanced_hidden_states                     │
└─────────────────────────────────────────────────────────┘

... (repeat for layers 2-36)

Final Output: "Recursion in Python is when a function calls 
itself. It needs a base case to stop. Example: factorial..."
```

---

## 📊 Comparison Table

| Aspect | Internal Memory (LM2) | External Memory (FAISS) |
|--------|----------------------|------------------------|
| **What's Stored** | Learned abstract patterns | Actual hidden states from past |
| **Format** | Matrix [16, 16] | Vectors [N, 2048] |
| **Size** | Fixed (16 slots) | Variable (up to 1M tokens) |
| **Content** | Abstract representations | Concrete past representations |
| **Learning** | Learned during training | Retrieved from storage |
| **Update** | Every forward pass | Periodically added |
| **Persistence** | Batch-to-batch | Long-term (saved to disk) |
| **Purpose** | Short-term patterns | Long-term knowledge |
| **Speed** | Very fast (in-memory) | Fast (FAISS optimized) |
| **Memory** | ~1 KB | ~17 GB (at full capacity) |

---

## 💡 Key Insights

### Internal Memory:
- ✅ Learns **what patterns matter** (not specific content)
- ✅ Adapts **dynamically** to current task
- ✅ Stores **abstract concepts** (like "subject", "action", "reference")
- ✅ Very **lightweight** and **fast**

### External Memory:
- ✅ Stores **actual past content** (specific hidden states)
- ✅ Retrieves **relevant examples** from history
- ✅ Provides **concrete context** from past sequences
- ✅ Scales to **millions of tokens**

### Together:
🚀 Internal memory provides **learned intuition**  
🚀 External memory provides **factual recall**  
🚀 Combined: **Best of both worlds!**

---

## 🔧 Configuration

### Control What Gets Stored:

```yaml
# configs/train.yaml

model:
  # Internal Memory
  use_internal_memory: true
  memory_slots: 16          # How many slots to learn
  num_mem_heads: 4          # Attention heads for memory
  
  # External Memory
  use_external_memory: true
  external_memory_size: 1048576  # Max tokens to store
  retrieval_k: 8            # How many to retrieve
  chunk_size: 4             # Granularity of storage
  
train:
  external_memory_update_freq: 100  # Add to FAISS every N steps
```

---

## 🎯 Practical Examples

### Example 1: Code Generation

**Internal Memory Learns:**
- Slot 1: Indentation patterns
- Slot 2: Function definition structure
- Slot 3: Variable naming conventions
- Slot 4: Import statement patterns

**External Memory Stores:**
- Previous function definitions
- Common code patterns
- Library usage examples
- Error handling snippets

### Example 2: Conversation

**Internal Memory Learns:**
- Slot 1: Current topic
- Slot 2: User's intent
- Slot 3: Conversation state
- Slot 4: Politeness level

**External Memory Stores:**
- Previous user questions
- Past responses
- Context from earlier in conversation
- Related topics discussed

---

This explains exactly what data lives where and how it's used! 🎯
