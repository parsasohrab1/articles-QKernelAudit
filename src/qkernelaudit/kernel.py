from __future__ import annotations

import numpy as np


def _hadamard_all(psi: np.ndarray, n: int) -> np.ndarray:
    """Apply H on every qubit to a batch of statevectors, shape (batch, 2**n)."""
    b = psi.shape[0]
    out = psi.reshape(b, *([2] * n))
    for ax in range(1, n + 1):
        a0 = np.take(out, 0, axis=ax)
        a1 = np.take(out, 1, axis=ax)
        out = np.stack([(a0 + a1), (a0 - a1)], axis=ax) / np.sqrt(2)
    return out.reshape(b, 2**n)


def zz_phases(X: np.ndarray) -> np.ndarray:
    """Diagonal phase exp(i * (sum_i x_i z_i + sum_{i<j} (pi - x_i)(pi - x_j) z_i z_j)) of the ZZ feature map.

    X: (m, n) features; returns (m, 2**n) complex phases for all computational basis states.
    """
    n = X.shape[1]
    idx = np.arange(2**n)
    z = 1 - 2 * (
        (idx[:, None] >> (n - 1 - np.arange(n))) & 1
    )  # (2**n, n) of +-1, qubit 0 = most significant bit
    th = X @ z.T  # (m, 2**n)
    for i in range(n):
        for j in range(i + 1, n):
            th = th + (np.pi - X[:, [i]]) * (np.pi - X[:, [j]]) * (z[:, i] * z[:, j])[None, :]
    return np.exp(1j * th)


def feature_states(X: np.ndarray, reps: int = 2) -> np.ndarray:
    """|phi(x)> = (D(x) H^n)^reps |0...0>."""
    m, n = X.shape
    psi = np.zeros((m, 2**n), dtype=complex)
    psi[:, 0] = 1.0
    D = zz_phases(X)
    for _ in range(reps):
        psi = D * _hadamard_all(psi, n)
    return psi


def quantum_kernel(A: np.ndarray, B: np.ndarray | None = None, reps: int = 2) -> np.ndarray:
    pa = feature_states(A, reps)
    pb = pa if B is None else feature_states(B, reps)
    return np.abs(pa.conj() @ pb.T) ** 2


def concentration(K: np.ndarray) -> dict[str, float]:
    iu = np.triu_indices_from(K, 1)
    off = K[iu]
    return {
        "mean_offdiag": float(off.mean()),
        "var_offdiag": float(off.var()),
        "min_offdiag": float(off.min()),
    }


def target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    s = 2.0 * y - 1.0
    Y = np.outer(s, s)
    Kc = K - K.mean()
    return float((Kc * Y).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Y) + 1e-12))
