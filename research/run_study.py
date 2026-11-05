"""Quantum ZZ-feature-map kernel (exact simulation) versus classical SVMs on real tabular data.

usage: python research/run_study.py <wdbc.data> <heart_failure_clinical_records_dataset.csv>
Data: Wisconsin Diagnostic Breast Cancer (UCI, n=569) and Heart Failure Clinical Records (UCI, n=299, 'time' excluded).
PCA to q components (q qubits), min-max scaling to [0, 1], multiplied by a bandwidth; 5-fold CV, 2 repeats;
hyper-parameters (C, bandwidth, RBF gamma) chosen by inner 3-fold CV on the training part only.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from qkernelaudit.kernel import concentration, quantum_kernel, target_alignment

warnings.filterwarnings("ignore")
QUBITS = (2, 4, 6, 8, 10)
BANDWIDTHS = (0.1, 0.3, 1.0)
CS = (0.1, 1.0, 10.0)


def load(wdbc: str, hf: str):
    d = pd.read_csv(wdbc, header=None)
    out = {"wdbc": (d.iloc[:, 2:].to_numpy(float), (d[1] == "M").astype(int).to_numpy())}
    h = pd.read_csv(hf)
    out["heart_failure"] = (
        h.drop(columns=["DEATH_EVENT", "time"]).to_numpy(float),
        h["DEATH_EVENT"].to_numpy(),
    )
    return out


def classical(Xtr, ytr, Xte, yte):
    res = {}
    for name, kern, grid in (
        ("linear", "linear", {"C": CS}),
        ("rbf", "rbf", {"C": CS, "gamma": (0.01, 0.03, 0.1, 0.3, 1.0)}),
    ):
        gs = GridSearchCV(
            SVC(kernel=kern), grid, cv=StratifiedKFold(3, shuffle=True, random_state=0)
        ).fit(Xtr, ytr)
        res[name] = float(gs.score(Xte, yte))
    return res


def quantum(Ztr, ytr, Zte, yte, reps=2):
    best = None
    for bw in BANDWIDTHS:
        Ktr = quantum_kernel(Ztr * bw, reps=reps)
        gs = GridSearchCV(
            SVC(kernel="precomputed"),
            {"C": CS},
            cv=StratifiedKFold(3, shuffle=True, random_state=0),
        ).fit(Ktr, ytr)
        if best is None or gs.best_score_ > best[0]:
            best = (gs.best_score_, bw, gs.best_params_["C"], Ktr)
    _, bw, C, Ktr = best
    clf = SVC(kernel="precomputed", C=C).fit(Ktr, ytr)
    acc = float(clf.score(quantum_kernel(Zte * bw, Ztr * bw, reps=reps), yte))
    return acc, bw, {**concentration(Ktr), "alignment": target_alignment(Ktr, ytr)}


def main(wdbc: str, hf: str):
    data = load(wdbc, hf)
    out = {
        "note": "exact statevector simulation of a ZZ feature map (2 repetitions); real UCI data; 5-fold CV x 2",
        "datasets": {},
    }
    for dname, (X, y) in data.items():
        out["datasets"][dname] = {"n": len(y), "by_qubits": {}}
        for q in QUBITS:
            accs = {"quantum": [], "linear": [], "rbf": []}
            conc = []
            bws = []
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
            for tr, te in cv.split(X, y):
                sc = StandardScaler().fit(X[tr])
                pca = PCA(q, random_state=0).fit(sc.transform(X[tr]))
                Ptr, Pte = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[te]))
                mm = MinMaxScaler().fit(Ptr)
                Ztr, Zte = np.clip(mm.transform(Ptr), 0, 1), np.clip(mm.transform(Pte), 0, 1)
                a, bw, c = quantum(Ztr, y[tr], Zte, y[te])
                accs["quantum"].append(a)
                conc.append(c)
                bws.append(bw)
                cl = classical(Ptr, y[tr], Pte, y[te])
                accs["linear"].append(cl["linear"])
                accs["rbf"].append(cl["rbf"])
            out["datasets"][dname]["by_qubits"][str(q)] = {
                "accuracy_mean": {k: float(np.mean(v)) for k, v in accs.items()},
                "accuracy_sd": {k: float(np.std(v)) for k, v in accs.items()},
                "kernel_mean_offdiag": float(np.mean([c["mean_offdiag"] for c in conc])),
                "kernel_var_offdiag": float(np.mean([c["var_offdiag"] for c in conc])),
                "kernel_alignment": float(np.mean([c["alignment"] for c in conc])),
                "chosen_bandwidths": {str(b): bws.count(b) for b in sorted(set(bws))},
            }
            v = out["datasets"][dname]["by_qubits"][str(q)]
            print(
                dname,
                q,
                {k: round(x, 3) for k, x in v["accuracy_mean"].items()},
                "var",
                f"{v['kernel_var_offdiag']:.2e}",
                "mean",
                round(v["kernel_mean_offdiag"], 4),
                v["chosen_bandwidths"],
            )
    # concentration audit on the full data sets: variance of off-diagonal kernel values versus qubits at fixed bandwidths
    out["concentration_full_data"] = {}
    for dname, (X, y) in data.items():
        out["concentration_full_data"][dname] = {}
        for q in QUBITS:
            P = PCA(q, random_state=0).fit_transform(StandardScaler().fit_transform(X))
            Z = MinMaxScaler().fit_transform(P)
            out["concentration_full_data"][dname][str(q)] = {
                str(bw): concentration(quantum_kernel(Z * bw)) for bw in (0.1, 0.3, 1.0, 3.0)
            }
        print(
            dname,
            {
                q: {bw: f"{v['var_offdiag']:.1e}" for bw, v in d.items()}
                for q, d in out["concentration_full_data"][dname].items()
            },
        )
    Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
