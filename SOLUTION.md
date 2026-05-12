# SOLUTION.md - SMILES-2026 Hallucination Detection

## 1. Reproducibility Instructions

### Required Environment

- Python 3.10+
- A CUDA-enabled GPU is highly recommended for efficient hidden-state extraction.
- The solution was developed and tested on Google Colab using a T4 GPU.

No additional external libraries are required beyond the original project requirements.

### Exact Commands to Run

To reproduce the final submission and generate the same `predictions.csv`, run the following commands:

```bash
git clone https://github.com/avmera/hallucination-detection.git
cd hallucination-detection
pip install -r requirements.txt
python solution.py
```

After running `python solution.py`, the following files should be generated:

```text
predictions.csv
results.json
```

The generated `predictions.csv` is the file submitted through the application form.

### Important Implementation Details

The repository is self-contained and runnable with the provided `solution.py` file. The solution does not require changes to the fixed infrastructure files.

The final implementation depends on enabling geometric features. In `solution.py` or the calling script, the aggregation function must be executed with:

```python
USE_GEOMETRIC = True
```

This is critical because the final score depends on extracting the representation drift and cosine similarity features from `extract_geometric_features`.

The repository should contain the following files:

```text
aggregation.py
probe.py
splitting.py
solution.py
evaluate.py
model.py
requirements.txt
results.json
SOLUTION.md
```

The application form also requires a public link to the generated `predictions.csv` file.

---

## 2. Final Solution Description

### What Components Were Modified?

I exclusively modified `aggregation.py` to handle advanced feature extraction and feature engineering.

Both `probe.py` and `splitting.py` were kept at their default implementations:

- `probe.py`: default binary MLP classifier
- `splitting.py`: default train/validation/test split strategy

This means the final solution focuses on improving the hidden-state representation rather than changing the classifier or the evaluation split.

### Final Approach

The core philosophy of my approach relies on mechanistic interpretability and tracking the information flow within the Qwen2.5-0.5B model.

Instead of relying on a complex classifier, I provided a highly refined and low-dimensional signal to the default MLP probe.

The final approach consists of two main components:

1. Late-layer mean pooling
2. Representation drift through geometric features

---

### 2.1 Late-Layer Mean Pooling

In the `aggregate` function, instead of using only the final hidden state, I sliced the last 6 layers of the final token and applied mean pooling:

```python
late_layers = hidden_states[-6:, last_pos, :]
feature = late_layers.mean(dim=0)
```

This smooths out syntactic noise and captures a more stable “Truth Direction” vector.

The reason for using the final token is that Qwen2.5-0.5B is a causal decoder-only model, so the final token representation has access to the previous context. In my experiments, the last-token representation performed better than token-level mean pooling.

The reason for using the last 6 layers is that late layers are more closely related to the model’s final response behavior. Averaging them makes the representation more stable than relying on only one final layer.

---

### 2.2 Representation Drift: Geometric Features

In `extract_geometric_features`, I introduced behavioral indicators of hallucination by measuring how the hidden representation changes across layers.

The geometric features include:

#### Inter-layer Cosine Similarity

I computed cosine similarity between all consecutive layers, from layer `L` to layer `L+1`.

This measures representation drift or hesitation.

The intuition is that truthful answers tend to stabilize earlier, while hallucinated answers may cause continuous shifts in later layers.

#### Late-layer L2 Norms

I extracted L2 norms from the late-layer representations.

These features capture anomalous activation magnitude or spikes that may appear when the model fabricates facts.

#### Layer of Maximum Change

I explicitly calculated the layer where the model representation shifted the most.

This is computed as the index of the minimum cosine similarity:

```python
max_change_layer = torch.argmin(cos_sim)
```

This feature helps the classifier identify where the strongest representation change happened.

#### Sequence Length

I also included sequence length as a simple additional signal.

Hallucinated answers can sometimes be unusually long or short, so this feature may provide useful supporting information.

---

### Why These Choices?

The training dataset is extremely small, with only 689 samples, compared to the high-dimensional hidden states of 896 dimensions.

Modifying the classifier in `probe.py` to a more complex architecture risks severe overfitting. For this reason, I intentionally kept the probe simple and shifted the focus to mathematical feature engineering in `aggregation.py`.

By injecting geometric knowledge about how language models behave when uncertain, I shifted the burden from the classifier’s weights to mathematically grounded features.

This allowed the default MLP probe to receive explicit indicators of hesitation and representation drift without adding unnecessary learnable parameters.

---

### What Contributed Most to Improving the Metric?

Appending the geometric features from `extract_geometric_features` contributed the most to improving the metric.

Specifically, the inter-layer cosine similarity trajectory combined with the layer of maximum change was the biggest contributor.

This helped because it explicitly fed the probe information about how the language model arrived at its answer, separating truthful stability from hallucinated instability.

In my experiments, this safely pushed the metric up to around 74.

The final direction was:

```text
Late-layer final-token representation
+
Geometric representation-drift features
```

---

## 3. Experiments and Failed Attempts

During the competition, several alternative approaches were tested but ultimately discarded.

### 3.1 Complex Probe Architectures

**Idea:**

I experimented with modifying `probe.py` to include heavier architectures, such as:

- Deeper MLPs
- Support Vector Machines (SVM)
- Custom attention mechanisms to weigh the importance of different layers

**Why discarded:**

While these models achieved near-perfect scores on the training data, they collapsed on the validation and test sets due to extreme overfitting.

This happened because of the curse of dimensionality: the hidden-state feature dimension is large compared with the number of samples.

The dataset contains only 689 samples, while the hidden representation has 896 dimensions. Increasing the classifier complexity made the model memorize the training fold instead of learning a robust hallucination signal.

Reverting to the default simple MLP proved much more robust when supplied with the right engineered features.

---

### 3.2 Topological Data Analysis and Full Sequences

**Idea:**

I attempted to extract topological features and analyze the hidden states across the entire sequence of generated tokens, instead of focusing only on the final token.

This included topology-inspired ideas such as:

- Pairwise distances between hidden-state representations
- Structural summaries of representation trajectories
- Full-sequence hidden-state patterns

**Why discarded:**

The resulting feature vectors introduced too much dimensionality and syntactic noise.

The baseline probe struggled to isolate the actual hallucination signal from the structural and topological noise, which caused the overall score to drop.

For this reason, I discarded the full topological feature approach from the final solution.

---

### 3.3 First-Token vs. Last-Token Representation

**Idea:**

I tried extracting features from the very first generated token, which can be interpreted as an early “decision” token.

**Why discarded:**

While this idea was theoretically reasonable, it performed worse in practice.

Because Qwen2.5-0.5B uses a causal, decoder-only architecture, the last token serves as the ultimate contextual summary of the entire generated sequence.

The first token lacked the aggregated context needed to reliably detect hallucinations that may appear later in the response.

The last token provided a consistently better baseline score than early-token features.

---

## 4. Final Summary

My final solution focuses on feature engineering rather than classifier complexity.

I modified only `aggregation.py`, while keeping both `probe.py` and `splitting.py` at their default implementations.

The final method uses:

```text
aggregation.py:
    Last real token
    Mean pooling over the last 6 layers
    Inter-layer cosine similarity
    Late-layer L2 norms
    Layer of maximum change
    Sequence length

probe.py:
    Default implementation

splitting.py:
    Default implementation
```

This approach was selected because it improved the metric while keeping the solution simple, lightweight, and reproducible.

The repository can be cloned and run using:

```bash
git clone https://github.com/avmera/hallucination-detection.git
cd hallucination-detection
pip install -r requirements.txt
python solution.py
```
