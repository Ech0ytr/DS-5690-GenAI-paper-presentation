# GLU Variants Improve Transformer - Paper Presentation

## Paper Overview

**Title**: GLU Variants Improve Transformer  
**Author**: Noam Shazeer (Google)  
**Year**: 2020  
**Paper Link**: [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)

### Summary
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

## First Question: Why is activation applied to W (gate) and not V (value)?

<details>
  <summary>Show Answer</summary>

### Architecture Behavior

In GLU-based feed-forward networks, the input is split into two parallel projections: W (gate) and V (value). These serve distinct roles in the computation.

### Functional Roles

- **W (Gate)**:  
  The gate projection is responsible for filtering and importance scoring. It determines how much of each feature should pass through to the output. Applying an activation function like GELU or Swish here helps shape the gating signal, introducing non-linearity and allowing the model to selectively amplify or suppress features.

- **V (Value)**:  
  The value projection carries the raw feature representation. It is meant to preserve the full richness of the learned features, including negative values, large magnitudes, and subtle variations. Applying an activation here would distort or suppress these features, reducing the model’s expressive capacity.

### Design Implication

By applying activation only to the gate, the GLU mechanism maintains a balance between control and expressiveness. The gate modulates the flow, while the value retains the full feature signal. This separation is key to GLU’s improved performance over standard FFNs.

</details>

---

## Second Question: How do the parameter dimensions and internal structure differ between the baseline linear FFN and the GLU-based FFN?

<details>
  <summary>Show Answer</summary>

### Architecture Comparison

**Standard FFN (Feed-Forward Network):**
- Input: `[768]`
- First Linear Layer: `W₁ [768×3072]`
- Hidden Activation: `[3072]` → ReLU
- Second Linear Layer: `W₂ [3072×768]`
- Output: `[768]`

**GLU-based FFN:**
- Input: `[768]`
- Two Parallel Projections:
  - Gate: `W [768×2048]`
  - Value: `V [768×2048]`
- Activation: GLU (element-wise multiplication after activation)
- Combined Hidden: `[2048]`
- Second Linear Layer: `W₂ [2048×768]`
- Output: `[768]`

### Structural Differences

- **Projection Strategy**:  
  The standard FFN uses a single wide projection to 3072 dimensions. GLU splits the input into two separate projections, each to 2048 dimensions, which are then combined via element-wise multiplication.

- **Activation Mechanism**:  
  In the standard FFN, ReLU is applied to the entire hidden layer. In GLU, activation is applied only to the gate path before combining with the value path, introducing a dynamic gating mechanism.

- **Parameter Count**:  
  GLU adds a second projection (V), increasing the number of parameters in the first layer. However, the hidden size is reduced from 3072 to 2048, which partially offsets the increase.

- **Dimensional Consistency**:  
  Both architectures preserve the input and output dimensions of `[768]`, ensuring compatibility with the Transformer block. The internal structure is what differs.

### Design Implication

This architectural shift allows GLU variants to learn more selective and expressive representations. The gating mechanism enhances control over feature flow, contributing to lower perplexity and improved generalization in downstream tasks.

</details>

---

## Algorithmic Comparison

### **Algorithm 1: Standard Feed-Forward Network (Baseline)**
```
ALGORITHM: StandardFFN
INPUT: x ∈ ℝ^(batch_size × seq_len × d_model)  // Input tensor from attention layer
PARAMETERS: 
    W₁ ∈ ℝ^(d_model × d_ff)  // First projection matrix (768 × 3072)
    W₂ ∈ ℝ^(d_ff × d_model)  // Second projection matrix (3072 × 768)
OUTPUT: y ∈ ℝ^(batch_size × seq_len × d_model)  // Output tensor

PROCEDURE:
1: h ← MatMul(x, W₁)           // Project to higher dimension: [B, L, 3072]
2: h ← ReLU(h)                  // Apply activation: max(0, h)
3: y ← MatMul(h, W₂)           // Project back: [B, L, 768]
4: RETURN y

COMPLEXITY: O(2 × batch_size × seq_len × d_model × d_ff)
```

### **Algorithm 2: GLU Feed-Forward Network (Proposed)**
```
ALGORITHM: GLU_FFN
INPUT: x ∈ ℝ^(batch_size × seq_len × d_model)  // Input tensor from attention layer
PARAMETERS:
    W ∈ ℝ^(d_model × d_ff_glu)   // Gate projection matrix (768 × 2048)
    V ∈ ℝ^(d_model × d_ff_glu)   // Value projection matrix (768 × 2048)
    W₂ ∈ ℝ^(d_ff_glu × d_model)  // Output projection matrix (2048 × 768)
    activation ∈ {sigmoid, swish, gelu, relu, identity}  // Activation function
OUTPUT: y ∈ ℝ^(batch_size × seq_len × d_model)  // Output tensor

PROCEDURE:
1: gate ← MatMul(x, W)          // Compute gate path: [B, L, 2048]
2: value ← MatMul(x, V)         // Compute value path: [B, L, 2048]
3: gate ← activation(gate)      // Apply activation to gate only
4: h ← gate ⊙ value            // Element-wise multiplication
5: y ← MatMul(h, W₂)           // Project to output: [B, L, 768]
6: RETURN y

COMPLEXITY: O(3 × batch_size × seq_len × d_model × d_ff_glu)
WHERE: d_ff_glu = (2/3) × d_ff to maintain parameter parity
```


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


### Why 1.677 → 1.633 Matters

**Converting to Real Performance:**
- **Baseline (1.677)**: ~59.6% average confidence per prediction
- **GEGLU (1.633)**: ~61.2% average confidence per prediction
- **Impact**: This 2.7% relative improvement compounds exponentially over sequences

**The Compounding Effect:**

When generating 100 tokens, the probability of maintaining coherence:

| Model | Per-token Confidence | 100-token Coherence | Relative Improvement |
|-------|---------------------|--------------------|--------------------|
| Baseline | 0.596 | 1.4 × 10⁻²³ | - |
| GEGLU | 0.612 | 7.8 × 10⁻²² | **55× better** |

> **Key Insight**: GEGLU is 55× more likely to generate coherent long-form text

This foundational improvement in language modeling directly translates to better performance across all downstream tasks - which is why modern LLMs like GPT-3, PaLM, and LLaMA have all adopted GLU variants.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1247accd-cfa4-4f81-bfb2-52e8e9da09fe" width="45%" alt="Left Image" />
  <img src="https://github.com/user-attachments/assets/140d75ae-c579-4d24-b81a-5bcd5ec65573" width="45%" alt="Right Image" />
</p>

---

### Key Insights from Visualizations

**Left Graph - Perplexity Comparison**:
- GEGLU and SwiGLU achieve the lowest perplexity (1.633, 1.636)
- Even simple variants (Bilinear, ReGLU) outperform standard activations
- Clear performance grouping: GLU variants > Standard activations

**Right Graph - Compound Effect**:
- Small per-token improvements lead to exponential differences over sequences
- At 100 tokens, GEGLU is 55× more likely to maintain coherence
- The gap widens dramatically for longer generations (typical for real applications)

---


## How GLU Works: The Two-Path Architecture

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

## Real-World Impact

### Who's Using SwiGLU Now?

| Model | Company | Year | Uses GLU Variant |
|-------|---------|------|------------------|
| GPT-3 | OpenAI | 2020 | ✅ SwiGLU |
| PaLM | Google | 2022 | ✅ SwiGLU |
| LLaMA | Meta | 2023 | ✅ SwiGLU |
| Gemma | Google | 2024 | ✅ GeGLU |
| Claude | Anthropic | - | ✅ (variant) |

### Influence on Transformer Design and Adoption

This work has reshaped how researchers think about the internal architecture of Transformer models. By demonstrating that GLU variants—especially GEGLU and SwiGLU—can significantly improve pretraining perplexity and downstream performance, the paper shifted attention toward activation functions as a meaningful site of innovation. These findings have influenced the design of major language models such as GPT-3, PaLM, and LLaMA, which now incorporate GLU-based activations in their feed-forward blocks. The result is a broader recognition that small architectural changes can yield substantial gains in coherence, efficiency, and generalization, especially when compounded over long sequences.

### Position Within the Broader AI Research Landscape

The introduction of GLU variants builds on earlier ideas from gated mechanisms in LSTMs and convolutional networks, extending them into the Transformer’s feed-forward layers. It complements past work focused on scaling laws and attention mechanisms by showing that activation design also plays a critical role in model quality. In the present, GLU variants have become a standard component in high-performing LLMs, contributing to their fluency and robustness across tasks. Looking ahead, this work opens the door to further exploration of adaptive or learnable activations, and encourages deeper investigation into how subtle architectural choices affect interpretability, training dynamics, and cross-domain generalization.

---


## Critical Analysis

### What Was Overlooked

- **Interpretability and Mechanistic Insight**:  
  The authors do not analyze *why* GLU variants outperform traditional activations. There is no investigation into how these activations affect:
  - Attention distributions
  - Gradient flow and saturation
  - Representation sparsity or expressiveness

  Without this, the paper lacks a mechanistic explanation for the observed performance gains.

- **Robustness and Generalization**:  
  The experiments focus on standard benchmarks (e.g., GLUE, WMT), but do not test:
  - Low-resource settings
  - Adversarial inputs
  - Cross-domain generalization

  This limits confidence in how GLU variants perform under real-world variability.

---

### What Could Be Developed Further

- **Activation Dynamics Analysis**:  
  Future work could include probing studies or visualization techniques (e.g., attention heatmaps, activation histograms) to understand how GLU variants shape internal model behavior.

- **Robustness Benchmarks**:  
  Evaluating GLU-based models on tasks like adversarial QA, noisy translation, or zero-shot generalization would help assess their reliability.

- **Efficiency Trade-offs**:  
  The paper does not discuss computational cost or training stability. A deeper look into FLOPs, convergence speed, and memory usage would help practitioners decide when to adopt GLU variants.


The Paper says
> "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."

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

