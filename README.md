# articles-QKernelAudit

Audit of a quantum feature-map kernel (ZZ feature map, 2 repetitions, exact statevector simulation, 2–10 qubits) against linear and
RBF support-vector machines on **real tabular data** (UCI Wisconsin Diagnostic Breast Cancer and Heart Failure Clinical Records),
including kernel concentration versus qubit count. No quantum hardware. TRL 4.

```
pip install -e ".[dev]" && python -m pytest
python research/run_study.py <wdbc.data> <heart_failure_clinical_records_dataset.csv>
```
