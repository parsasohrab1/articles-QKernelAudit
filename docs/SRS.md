# Software Requirements Specification – QKernelAudit

- SR-1 `kernel.feature_states`: |phi(x)> = (D(x) H^n)^reps |0>, D the diagonal ZZ-feature-map phase exp(i(sum x_i z_i + sum_{i<j} (pi - x_i)(pi - x_j) z_i z_j)); exact statevector in numpy (qubit 0 most significant bit). A unit test compares the 2-qubit state with an explicit matrix construction.
- SR-2 `kernel.quantum_kernel`: |<phi(a)|phi(b)>|^2; tests check symmetry, unit diagonal, positive semi-definiteness and range.
- SR-3 `kernel.concentration` (mean and variance of off-diagonal entries), `kernel.target_alignment`.
- SR-4 `research/run_study.py`: PCA to q components (fit on the training fold), min–max scaling to [0, 1], bandwidth in {0.1, 0.3, 1.0}, C in {0.1, 1, 10} chosen by inner 3-fold CV; linear and RBF SVMs tuned in the same way on the same PCA features; 5-fold CV repeated twice.
