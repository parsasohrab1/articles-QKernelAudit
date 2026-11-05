# TRL assessment – QKernelAudit

**Claimed level: TRL 4** (simulation-based validation on public real data). Not TRL 5: no hardware, no shot noise, two small data sets.

## Cross-validated accuracy (mean ± sd over 10 folds), by number of qubits = PCA components

Wisconsin Diagnostic Breast Cancer (n = 569):

| Qubits | Quantum kernel | Linear SVM | RBF SVM | Off-diagonal kernel variance | Alignment | Chosen bandwidths |
|---|---|---|---|---|---|---|
| 2 | 0.931 ± 0.019 | 0.951 ± 0.017 | 0.941 ± 0.022 | 9.1e-02 | 0.29 | {'0.3': 1, '1.0': 9} |
| 4 | 0.954 ± 0.020 | 0.962 ± 0.020 | 0.959 ± 0.020 | 2.2e-02 | 0.24 | {'0.1': 5, '0.3': 5} |
| 6 | 0.962 ± 0.015 | 0.966 ± 0.018 | 0.967 ± 0.014 | 2.7e-02 | 0.23 | {'0.1': 7, '0.3': 3} |
| 8 | 0.965 ± 0.016 | 0.970 ± 0.020 | 0.972 ± 0.009 | 2.9e-02 | 0.23 | {'0.1': 10} |
| 10 | 0.962 ± 0.012 | 0.977 ± 0.016 | 0.977 ± 0.016 | 4.8e-02 | 0.21 | {'0.1': 10} |

Heart Failure Clinical Records (n = 299):

| Qubits | Quantum kernel | Linear SVM | RBF SVM | Off-diagonal kernel variance | Alignment | Chosen bandwidths |
|---|---|---|---|---|---|---|
| 2 | 0.704 ± 0.026 | 0.692 ± 0.016 | 0.697 ± 0.017 | 7.4e-02 | 0.01 | {'0.1': 1, '1.0': 9} |
| 4 | 0.719 ± 0.022 | 0.711 ± 0.027 | 0.727 ± 0.036 | 2.8e-02 | 0.08 | {'0.1': 4, '0.3': 6} |
| 6 | 0.707 ± 0.036 | 0.741 ± 0.032 | 0.721 ± 0.037 | 1.5e-02 | 0.09 | {'0.1': 10} |
| 8 | 0.704 ± 0.039 | 0.741 ± 0.051 | 0.727 ± 0.036 | 3.1e-02 | 0.09 | {'0.1': 10} |
| 10 | 0.704 ± 0.029 | 0.724 ± 0.042 | 0.719 ± 0.035 | 3.7e-02 | 0.10 | {'0.1': 10} |

## Kernel concentration (variance of off-diagonal kernel entries, full data, by bandwidth)

WDBC:

| Qubits | 0.1 | 0.3 | 1.0 | 3.0 |
|---|---|---|---|---|
| 2 | 1.9e-04 | 9.6e-03 | 1.0e-01 | 8.0e-02 |
| 4 | 2.5e-03 | 3.9e-02 | 3.0e-02 | 8.0e-03 |
| 6 | 1.1e-02 | 6.6e-02 | 5.0e-03 | 5.3e-04 |
| 8 | 2.6e-02 | 4.2e-02 | 9.9e-04 | 5.5e-05 |
| 10 | 4.8e-02 | 1.8e-02 | 1.7e-04 | 6.4e-06 |

Heart failure:

| Qubits | 0.1 | 0.3 | 1.0 | 3.0 |
|---|---|---|---|---|
| 2 | 3.0e-04 | 1.5e-02 | 8.7e-02 | 6.7e-02 |
| 4 | 3.4e-03 | 4.4e-02 | 1.7e-02 | 5.8e-03 |
| 6 | 1.3e-02 | 4.7e-02 | 2.2e-03 | 4.4e-04 |
| 8 | 3.0e-02 | 1.0e-02 | 2.5e-04 | 3.2e-05 |
| 10 | 3.4e-02 | 2.3e-03 | 4.1e-05 | 4.1e-06 |

## Findings and caveats
- **The quantum kernel does not beat the best tuned classical SVM at any setting except heart failure with 2 qubits (+0.7 points over the RBF SVM, far inside the fold standard deviation).** On WDBC it is 0.4–2 points below the best classical SVM at every qubit count (1.5 points at 10 qubits: 0.962 against 0.977); on heart failure it is 0.8–3.7 points below from 4 qubits on. Fold standard deviations are 1–2 points (WDBC) and 2–5 points (heart failure), so most single differences are not significant — what the tables show is the absence of any advantage, not a measured deficit.
- Kernel concentration is clearly visible: at bandwidth 1.0 the off-diagonal variance falls from 10^-1 (2 qubits) to 10^-4 (10 qubits) on WDBC, and to 10^-6 at bandwidth 3.0. The tuning escapes it by choosing the smallest bandwidth (0.1) from 6 qubits on — the edge of the grid searched — where the kernel is close to a low-frequency, nearly classical map; wider grids were not explored.
- With ≥ 6 qubits the selected bandwidth is 0.1 in every fold, so a better result might lie below the grid.
- The kernel is evaluated exactly; on hardware every entry would need many shots, and the concentrated entries (variance 10^-4 and below) would need a shot count that grows accordingly — not simulated here.
- Two small data sets with PCA preprocessing; conclusions apply to these settings only.
