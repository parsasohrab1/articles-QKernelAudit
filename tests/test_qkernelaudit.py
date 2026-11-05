import numpy as np

from qkernelaudit.kernel import concentration, feature_states, quantum_kernel, target_alignment


def test_states_are_normalised_and_kernel_is_valid():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (12, 4))
    psi = feature_states(X)
    assert np.allclose(np.linalg.norm(psi, axis=1), 1.0)
    K = quantum_kernel(X)
    assert np.allclose(K, K.T) and np.allclose(np.diag(K), 1.0)
    assert np.linalg.eigvalsh(K).min() > -1e-9  # positive semi-definite
    assert (K >= -1e-12).all() and (K <= 1 + 1e-9).all()


def test_two_qubit_kernel_matches_direct_circuit_simulation():
    # independent check of qubit 0 = most significant bit, one rep, n = 2, by building the matrices explicitly
    x = np.array([[0.3, 0.9]])
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    H2 = np.kron(H, H)
    z = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    th = (
        x[0, 0] * z[:, 0]
        + x[0, 1] * z[:, 1]
        + (np.pi - x[0, 0]) * (np.pi - x[0, 1]) * z[:, 0] * z[:, 1]
    )
    psi = np.exp(1j * th) * (H2 @ np.array([1, 0, 0, 0]))
    assert np.allclose(feature_states(x, reps=1)[0], psi)


def test_concentration_and_alignment_basic():
    K = np.array([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1.0]])
    c = concentration(K)
    assert abs(c["mean_offdiag"] - 1.0 / 3) < 1e-9
    y = np.array([0, 0, 1])
    assert target_alignment(K, y) != 0
