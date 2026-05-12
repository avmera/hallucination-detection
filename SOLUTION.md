# SMILES-2026 Hallucination Detection Solution

## 1. Overview

This project addresses hallucination detection in language model responses. The goal is to classify each response as either truthful (`label = 0`) or hallucinated (`label = 1`) using hidden states extracted from Qwen2.5-0.5B.

I exclusively modified `aggregation.py` to handle advanced feature extraction and feature engineering. Both `probe.py` and `splitting.py` were kept at their default implementations.

The repository contains the full implementation and this report in Markdown format (`SOLUTION.md`). The solution is designed to be reproducible and runnable with the provided `solution.py` file.

---

## 2. Final Approach

The core philosophy of my approach relies on mechanistic interpretability and tracking the information flow within the Qwen2.5-0.5B model.

Instead of relying on a complex classifier, I provided a highly refined, low-dimensional signal to the default MLP probe. This was especially important because the dataset is small, with only 689 samples.

The final approach is based on two main components:

1. Late-layer mean pooling
2. Representation drift through geometric features

---

## 3. Late-Layer Mean Pooling

In `aggregate`, instead of just taking the last hidden state, I sliced the last 6 layers of the final token and applied mean pooling.

```python
late_layers = hidden_states[-6:, last_pos, :]
feature = late_layers.mean(dim=0)
```

This smooths out syntactic noise and captures a stable “Truth Direction” vector.

The reason for using the last token is that, in a causal language model, the final token representation has access to the previous context. In my experiments, the last-token representation produced stronger and more stable results than token-level mean pooling.

The reason for using the last 6 layers is that late layers are more closely related to the final response behavior. Averaging them makes the representation more stable than relying on only one final layer.

---

## 4. Representation Drift: Geometric Features

In `extract_geometric_features`, I introduced behavioral indicators of hallucination by measuring how the hidden representation changes across layers.

The geometric features include:

### 4.1 Inter-layer Cosine Similarity

I computed cosine similarity between all consecutive layers, from layer `L` to layer `L+1`.

This measures representation drift or hesitation.

The intuition is that truthful answers tend to stabilize earlier, while hallucinated answers may cause continuous shifts in later layers.

### 4.2 Late-layer L2 Norms

I extracted L2 norms from the late-layer representations.

These features capture anomalous activation magnitude or spikes that may appear when the model fabricates facts.

### 4.3 Layer of Maximum Change

I added an explicit scalar identifying the layer where the model representation shifted the most.

This is computed as the index of the minimum cosine similarity:

```python
max_change_layer = torch.argmin(cos_sim)
```

This feature helps the classifier understand where the strongest representation change happened.

### 4.4 Sequence Length

I also included the sequence length as a simple additional signal.

Hallucinated answers can sometimes be unusually long or short, so this feature may provide useful supporting information.

---

## 5. Why These Choices?

With only 689 samples, increasing the complexity of the classifier in `probe.py` can easily lead to severe overfitting.

For this reason, I kept the default probe and focused on improving the input features instead.

By injecting geometric knowledge about how language models behave when uncertain, I shifted the burden from the classifier weights to mathematically grounded features.

This keeps the solution lightweight and avoids making the classifier unnecessarily complex.

---

## 6. What Contributed Most?

The most important improvement came from appending `extract_geometric_features`, especially the cosine similarity trajectory across layers.

The cosine similarity trajectory helped capture representation drift and improved the metric compared with using only the aggregated hidden-state vector.

In my experiments, the final configuration safely pushed the model to around a 74 metric score.

The final direction was:

```text
Late-layer final-token representation
+
Geometric representation-drift features
```

---

## 7. Components Modified

### `aggregation.py`

This is the only file I modified for the final solution.

I changed it to:

- Use the last real non-padding token.
- Average the last 6 hidden layers in `aggregate`.
- Extract geometric representation-drift features in `extract_geometric_features`.
- Concatenate the aggregated feature vector with geometric features when `USE_GEOMETRIC=True`.

### `probe.py`

I kept `probe.py` at its default implementation.

The probe uses the default MLP classifier provided in the project.

### `splitting.py`

I kept `splitting.py` at its default implementation.

The split remains the original train/validation/test splitting strategy.

---

## 8. Experiments and Failed Attempts

During the competition, several alternative approaches were tested but ultimately discarded.

### 8.1 Complex Probe Architectures

I experimented with modifying `probe.py` to use heavier architectures, such as deeper MLPs, attention mechanisms, and SVM-style alternatives.

While these approaches showed near-perfect metrics on the training fold, they collapsed on the validation and test splits.

This happened because of the curse of dimensionality: the feature dimension is large compared with the number of samples.

Since there are 896 hidden dimensions but only 689 samples, increasing classifier complexity caused overfitting.

Reverting to the default simple MLP proved much more robust.

### 8.2 Topological Data Analysis and Deep Sequence Features

I attempted to map the full topological structure of the hidden states across all tokens.

However, the resulting feature vectors were too large and noisy.

The baseline probe could not separate the hallucination signal from the topological noise, which led to degraded performance.

For this reason, I discarded the full topological feature approach from the final solution.

### 8.3 First-Token vs. Last-Token Representation

I experimented with extracting features from the first generated token, which can be interpreted as an early “decision” token.

While this idea was theoretically reasonable, the last token performed better for the causal structure of Qwen2.5-0.5B.

The last token effectively acts as a summary of the entire context window, and it provided a consistently better baseline score than early-token features.

---

## 9. Reproducibility Instructions

The repository must be self-contained and runnable with the provided `solution.py` file.

Running `solution.py` should generate the submitted `predictions.csv`.

### 9.1 Required Environment

The solution was developed and tested on Google Colab using Python 3.

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

No additional external libraries are required beyond the original project requirements.

### 9.2 Required Files

The repository should contain the full implementation, including:

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

### 9.3 Exact Commands to Run

To reproduce the solution and generate `predictions.csv`, run:

```bash
pip install -r requirements.txt
python solution.py
```

After running the command, the following files should be generated:

```text
predictions.csv
results.json
```

The generated `predictions.csv` is the file submitted through the application form.

### 9.4 Important Implementation Details

The final implementation depends on enabling geometric features.

In `solution.py`, the final configuration should use:

```python
USE_GEOMETRIC = True
```

The final submitted configuration uses:

```text
aggregation.py:
    Last real token
    Mean pooling over the last 6 layers
    Geometric representation-drift features

probe.py:
    Default implementation

splitting.py:
    Default implementation
```

The solution does not require changes to the fixed infrastructure files.

---

## 10. Final Summary

My final solution focuses on feature engineering rather than classifier complexity.

I modified only `aggregation.py` and kept both `probe.py` and `splitting.py` at their default implementations.

The final method uses late-layer mean pooling on the last token and appends geometric representation-drift features. This approach was selected because it improved the metric while keeping the model simple, lightweight, and reproducible.
