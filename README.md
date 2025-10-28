# GLU Variants Improve Transformer - Paper Presentation

## Paper Overview

**Title**: GLU Variants Improve Transformer  
**Author**: Noam Shazeer (Google)  
**Year**: 2020  
**Paper Link**: [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)

### One-Line Summary
This paper shows that replacing standard ReLU-based feed-forward networks in Transformers with Gated Linear Unit (GLU) variants achieves consistent performance improvements across language tasks.

---

## Core Concept: What Problem Does This Solve?

### The Standard Transformer FFN 
In the original Transformer, each layer has two components:
1. **Multi-head Attention**: Captures relationships between tokens
2. **Feed-Forward Network (FFN)**: Processes each position independently

The standard FFN is simple:
```
Input (x) → Linear (W₁) → ReLU → Linear (W₂) → Output
```

**Formula**: `FFN(x) = max(0, xW₁)W₂`

### The Problem
- This design is simple but potentially suboptimal
- ReLU is a crude gating mechanism (if negative → zero)
- Limited expressiveness in feature transformation

### The Solution: Gated Linear Units (GLU)
Replace the simple FFN with a gating mechanism that uses **two parallel paths**:

```
Input (x) → Linear (W) → Activation ─┐
                                      × (element-wise) → Linear (W₂) → Output
Input (x) → Linear (V) ──────────────┘
```

**Formula**: `GLU_FFN(x) = (Activation(xW) ⊗ xV)W₂`

---

## Key Innovation: GLU Variants

The paper tests multiple activation functions in the gating path:

| Variant | Gate Activation | Formula | Perplexity Result |
|---------|----------------|---------|-------------------|
| **GLU** | Sigmoid | `(σ(xW) ⊗ xV)W₂` | 1.663 |
| **Bilinear** | None (linear) | `(xW ⊗ xV)W₂` | 1.648 |
| **ReGLU** | ReLU | `(max(0,xW) ⊗ xV)W₂` | 1.645 |
| **GEGLU** | GELU | `(GELU(xW) ⊗ xV)W₂` | **1.633** ⭐ |
| **SwiGLU** | Swish | `(Swish(xW) ⊗ xV)W₂` | **1.636** ⭐ |
| *Baseline* | *ReLU (standard)* | `max(0,xW₁)W₂` | *1.677* |

**Winner**: GEGLU and SwiGLU perform best!

---

## Understanding the Metrics

### What is Perplexity?
- **Definition**: A measure of how "confused" the model is when predicting text
- **Formula**: `Perplexity = e^(cross-entropy loss)`
- **Interpretation**: 
  - Lower is better
  - Perplexity of 1.0 = perfect prediction
  - Perplexity of 2.0 = as confused as a coin flip


### First Primary question: Why can perplexity be less than 2? Why does a change from 1.677 to 1.633 matter?
<details>
  <summary>Show Answer</summary>

### Why can perplexity be less than 2?

- Perplexity doesn't mean choosing between 1.677 items.
- It reflects the **equivalent uncertainty** of choosing **uniformly** among that many options.
- In reality:
  - Models predict from **50,000+ tokens**
  - But often with **high confidence** (e.g., 95% sure)
  - High confidence → **Low perplexity**

### Why does a change from 1.677 to 1.633 matter?

- These values represent **average uncertainty** across many predictions.
- **Converting to confidence**:
  - Perplexity **1.677** ≈ **59.6%** average confidence
  - Perplexity **1.633** ≈ **61.2%** average confidence
- **Impact**:
  - A **1.6% absolute improvement** may seem small, but it **compounds** over thousands or millions of predictions.
  - This can lead to **significantly better performance** in real-world tasks.

---

</details>


## 🔧 How GLU Works: The Two-Path Architecture

### The Parallel Paths Explained

**NOT splitting data** - same input goes through both paths:

```python
def SwiGLU_FFN(x):  # x shape: [batch, sequence_length, 768]
    # Path 1: Gate (decides "how much")
    gate = Swish(x @ W)     # W: [768, 2048] → outputs importance scores
    
    # Path 2: Value (provides "what")  
    value = x @ V           # V: [768, 2048] → outputs features
    
    # Combine via element-wise multiplication
    gated = gate * value    # [batch, sequence_length, 2048]
    
    # Project back to model dimension
    output = gated @ W₂     # W₂: [2048, 768]
    return output           # [batch, sequence_length, 768]
```

### Visual Architecture Comparison

#### Standard FFN:
```
Input [768] 
    ↓
   W₁ [768×3072]
    ↓
Hidden [3072]
    ↓
  ReLU
    ↓
   W₂ [3072×768]
    ↓
Output [768]
```

#### GLU FFN:
```
Input [768]
    ↓
   ┌─────────────┬─────────────┐
   ↓             ↓             
  W [768×2048]   V [768×2048]   
   ↓             ↓
Gate [2048]   Value [2048]
   ↓             ↓
Activation       ↓
   ↓             ↓
   └──────×──────┘ (element-wise)
         ↓
   Combined [2048]
         ↓
      W₂ [2048×768]
         ↓
    Output [768]
```

### Concrete Example: Processing "The cat sat on the mat"

For the word "cat":

**Gate Path (W)** learns to identify importance:
```
Output: [0.9, 0.2, 0.8, ...]
Meaning: [feature1: important, feature2: ignore, feature3: important]
```

**Value Path (V)** learns to extract features:
```
Output: [2.5, 1.8, 2.1, ...]
Meaning: [animal=2.5, living=1.8, pet=2.1]
```

**Combined Result**:
```
gate × value = [0.9×2.5, 0.2×1.8, 0.8×2.1] = [2.25, 0.36, 1.68]
Important features amplified, unimportant ones suppressed!
```

---

## Why GLU Works Better

### 1. **Multiplicative Interactions Create Non-linearity**

Even without activation functions, element-wise multiplication is non-linear:

```python
# Example showing non-linearity
x = [1, 2]
gate = [3, -1]  
value = [2, 4]
output = [3×2, -1×4] = [6, -4]

# If we double x:
x_doubled = [2, 4]
gate_doubled = [6, -2]
value_doubled = [4, 8]
output_doubled = [24, -16]  # 4x increase, not 2x! Non-linear!
```

**Why Non-linearity Matters**:
- Enables learning complex patterns
- Creates decision boundaries that aren't straight lines
- Allows conditional computation (if gate=0, output=0 regardless of value)

### 2. **Specialization Through Co-evolution**

During training through backpropagation:
- **W (Gate)** learns: "What information is important?"
- **V (Value)** learns: "What are the actual features?"
- Gradients create specialization pressure:
  - `∂L/∂W ∝ V × error` (W's gradient depends on V)
  - `∂L/∂V ∝ W × error` (V's gradient depends on W)
- They evolve together to divide labor efficiently

### 3. **Adaptive Computation**

Unlike ReLU (binary: on/off), GLU provides fine-grained control:
- Gate = 0 → completely block
- Gate = 0.5 → partially pass
- Gate = 1 → fully pass
- Gate > 1 → amplify (for some variants)

---

## Experimental Results

### Pre-training Results (Perplexity on C4 Dataset)

| Model | Perplexity @ 65k steps | Perplexity @ 524k steps | Improvement |
|-------|------------------------|-------------------------|-------------|
| **GEGLU** ⭐ | 1.942 (±0.004) | **1.633** | -2.6% |
| **SwiGLU** ⭐ | 1.944 (±0.010) | **1.636** | -2.5% |
| Bilinear | 1.960 (±0.005) | 1.648 | -1.7% |
| ReGLU | 1.953 (±0.003) | 1.645 | -1.9% |
| GLU | 1.982 (±0.006) | 1.663 | -0.8% |
| GELU | 1.983 (±0.005) | 1.679 | +0.1% |
| Swish | 1.994 (±0.003) | 1.683 | +0.4% |
| **ReLU (baseline)** | 1.997 (±0.005) | 1.677 | - |

### Fine-tuning Results

#### GLUE Benchmark (8 Language Understanding Tasks)

| Model | Average | CoLA | SST-2 | MRPC | STS-B | QQP | MNLI | QNLI | RTE |
|-------|---------|------|-------|------|-------|-----|------|------|-----|
| **ReGLU** | **84.67** | 56.16 | 94.38 | 92.06 | 89.97 | 88.86 | 86.20 | 92.68 | 81.59 |
| **SwiGLU** | **84.36** | 51.59 | 93.92 | 92.23 | **90.32** | **89.14** | 86.45 | **92.93** | 83.39 |
| GEGLU | 84.12 | 53.65 | 93.92 | 92.68 | 90.26 | 89.11 | 86.15 | 92.81 | 79.42 |
| GLU | 84.20 | 49.16 | **94.27** | 92.39 | 89.46 | 88.79 | **86.36** | 92.92 | **84.12** |
| Baseline | 83.80 | 51.32 | 94.04 | **93.08** | 89.64 | 89.01 | 85.83 | 92.81 | 80.14 |
| Original T5 | 83.28 | 53.84 | 92.68 | 92.07 | 88.02 | 88.67 | 84.24 | 90.48 | 76.28 |

#### SQuAD v1.1 (Question Answering)

| Model | Exact Match | F1 Score |
|-------|-------------|----------|
| **ReGLU** | 83.53 | **91.18** |
| **GEGLU** | **83.55** | 91.12 |
| Bilinear | 83.82 | 91.06 |
| SwiGLU | 83.42 | 91.03 |
| Baseline | 83.18 | 90.87 |
| Original T5 | 80.88 | 88.81 |

### Key Observations
- Consistent improvements across diverse tasks
- Different variants excel at different tasks
- Even "Bilinear" (no activation) beats baseline!
- Improvements are statistically significant (exceed 2× std dev)

---

## Parameter Efficiency Clarification

### GLU Maintains Same Parameter Count

**Standard FFN**: 2 matrices × 3072 hidden dims
```
W₁: [768 × 3072] = 2,359,296 params
W₂: [3072 × 768] = 2,359,296 params
Total: 4,718,592 parameters
```

**GLU FFN**: 3 matrices × 2048 hidden dims
```
W:  [768 × 2048] = 1,572,864 params (gate)
V:  [768 × 2048] = 1,572,864 params (value)
W₂: [2048 × 768] = 1,572,864 params (projection)
Total: 4,718,592 parameters
```

**The Trade-off**: 
- Standard: Fewer matrices (2) × Larger dimensions (3072)
- GLU: More matrices (3) × Smaller dimensions (2048)
- **Same total parameters, better performance!**

---

## Real-World Impact

### Who's Using SwiGLU Now?

| Model | Company | Year | Uses GLU Variant |
|-------|---------|------|------------------|
| GPT-3 | OpenAI | 2020 | ✅ SwiGLU |
| PaLM | Google | 2022 | ✅ SwiGLU |
| LLaMA | Meta | 2023 | ✅ SwiGLU |
| Gemma | Google | 2024 | ✅ GeGLU |
| Claude | Anthropic | - | ✅ (variant) |

### Why This Paper Matters
1. **Simple Change, Big Impact**: Just changing FFN → 2-4% improvement
2. **Thoroughly Validated**: Tested across 20+ different tasks
3. **Theoretically Interesting**: Shows importance of multiplicative interactions
4. **Practically Adopted**: Used in production by major LLMs
5. **Influenced Future Research**: Inspired gated attention and other multiplicative designs

---

## Metrics Explained

### Classification Metrics
- **Accuracy**: % of correct predictions
- **F1 Score**: Harmonic mean of precision and recall (good for imbalanced data)
- **MCC**: Matthews Correlation Coefficient (-1 to +1, handles imbalanced datasets)

### Regression Metrics
- **Pearson (PCC)**: Linear correlation between predicted and actual
- **Spearman (SCC)**: Rank correlation (robust to outliers)

### Exact Matching
- **EM**: % where prediction exactly matches ground truth
- **Strict but important**: "Paris, France" vs "Paris" = wrong

---

## Key Takeaways

### For ML Practitioners
1. **Architecture details matter** - Small changes can yield significant gains
2. **Gating mechanisms are powerful** - Adaptive computation beats fixed functions
3. **Systematic exploration pays off** - Testing variants revealed better designs
4. **Parameter count isn't everything** - Same parameters, better architecture = better results

### For Researchers
1. **Multiplicative interactions** > additive transformations
2. **Specialization emerges naturally** through gradient descent
3. **Simple baselines can be beaten** with thoughtful modifications
4. **Comprehensive evaluation is crucial** - Test on diverse tasks

### The Paper's Humorous Conclusion
> "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."

Despite this, the results speak for themselves!

---

## Future Directions

### Questions This Paper Raises
1. Why do GEGLU/SwiGLU specifically work best?
2. Could we stack multiple GLU layers?
3. What about gating in attention mechanisms?
4. Are there even better activation functions to discover?
5. Can we apply this to vision transformers?

### Subsequent Research Inspired
- **Flash Attention**: Uses gating ideas for efficient attention
- **Mixture of Experts**: Gates for routing between experts
- **State Space Models**: Mamba uses selective gating
- **Vision Transformers**: Adapted GLU for image processing

---

## Implementation Example

### Simple PyTorch Implementation of SwiGLU

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            # Standard FFN uses 4x hidden dim
            # We use 8/3x to maintain parameters with 3 matrices
            hidden_dim = int(dim * 8 / 3)
        
        self.w = nn.Linear(dim, hidden_dim, bias=False)
        self.v = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        
    def forward(self, x):
        # SwiGLU: (Swish(xW) ⊗ xV)W₂
        gate = F.silu(self.w(x))  # silu = Swish activation
        value = self.v(x)
        gated = gate * value  # element-wise multiplication
        output = self.w2(gated)
        return output

# Example usage
model = SwiGLU(dim=768)
x = torch.randn(32, 100, 768)  # [batch, sequence, features]
output = model(x)  # [32, 100, 768]
```

---

## References

### Main Paper
- Shazeer, N. (2020). **GLU Variants Improve Transformer**. arXiv:2002.05202

### Related Work
- Dauphin et al. (2016) - **Language Modeling with Gated Convolutional Networks** (Original GLU)
- Vaswani et al. (2017) - **Attention Is All You Need** (Original Transformer)
- Raffel et al. (2019) - **Exploring the Limits of Transfer Learning with T5**
- Hendrycks & Gimpel (2016) - **Gaussian Error Linear Units (GELUs)**
- Ramachandran et al. (2017) - **Searching for Activation Functions** (Swish)

---


## Summary: Why You Should Care

1. **Immediate Practical Value**: SwiGLU/GEGLU are used in production today
2. **Simple to Implement**: Just replace your FFN layers
3. **Consistent Gains**: Works across different model sizes and tasks
4. **Theoretical Insights**: Shows importance of multiplicative interactions
5. **Future Research**: Opens doors to exploring other gated architectures

---


