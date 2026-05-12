"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1). Called from ``solution.py``
via ``evaluate.run_evaluation``.

This version keeps the original baseline probe architecture, but adds
random seed control to improve reproducibility.
"""

from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


SEED = 42


def set_seed(seed: int = SEED) -> None:
    """
    Set random seeds to make training as reproducible as possible.

    Note:
        Exact reproducibility may still vary slightly across different
        hardware, CUDA versions, or PyTorch versions.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    This probe uses:
    - StandardScaler for feature normalization
    - A small MLP classifier
    - BCEWithLogitsLoss with class imbalance weighting
    - Validation-based threshold tuning
    """

    def __init__(self) -> None:
        super().__init__()

        # Fix random seed before building/training the model
        set_seed()

        self._net: nn.Sequential | None = None
        self._scaler = StandardScaler()
        self._threshold: float = 0.5

    def _build_network(self, input_dim: int) -> None:
        """Instantiate the network layers.

        Args:
            input_dim: Feature vector dimensionality.
        """

        # Fix seed before initializing neural network weights
        set_seed()

        self._net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``.

        Args:
            x: Float tensor of shape ``(n_samples, feature_dim)``.

        Returns:
            1-D tensor of raw pre-sigmoid logits.
        """
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )

        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``;
               0 = truthful, 1 = hallucinated.

        Returns:
            self
        """

        # Fix seed before training
        set_seed()

        X_scaled = self._scaler.fit_transform(X)

        self._build_network(X_scaled.shape[1])

        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        # Weight positive examples by neg/pos ratio to handle class imbalance.
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos

        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)],
            dtype=torch.float32,
        )

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
        )

        self.train()

        for _ in range(200):
            optimizer.zero_grad()

            logits = self(X_t)
            loss = criterion(logits, y_t)

            loss.backward()
            optimizer.step()

        self.eval()

        return self

    def fit_hyperparameters(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximize F1.

        Args:
            X_val: Validation feature matrix.
            y_val: Validation labels.

        Returns:
            self
        """

        probs = self.predict_proba(X_val)[:, 1]

        candidates = np.unique(
            np.concatenate(
                [
                    probs,
                    np.linspace(0.0, 1.0, 101),
                ]
            )
        )

        best_threshold = 0.5
        best_f1 = -1.0

        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(
                y_val,
                y_pred_t,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array with values in ``{0, 1}``.
        """

        probs = self.predict_proba(X)[:, 1]
        return (probs >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)``.
            Column 1 contains probability of hallucination.
        """

        X_scaled = self._scaler.transform(X)
        X_t = torch.from_numpy(X_scaled).float()

        self.eval()

        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).numpy()

        return np.stack(
            [
                1.0 - prob_pos,
                prob_pos,
            ],
            axis=1,
        )