"""Paso 3b: valida las etiquetas débiles (s03) contra los 58 estudios dorados.

Reporta, por etiqueta: precisión, recall, F1, exactitud, kappa de Cohen y AUC
del objetivo blando <L>_p. Compara dos políticas para las menciones inciertas
(U-ones: incierta = 1; U-zeros: incierta = 0), calcula la tasa de ruido
(1 - exactitud), la calibración de la confianza por tipo de evidencia y el
desglose por idioma (entregable 7).

Las reglas se congelan antes de correr este script: los dorados no se usan
para ajustarlas.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from common import LABELS, META, SEED

OUT = META.parents[1] / "reports"  # métricas agregadas: sí se versionan


def per_label(gold, pred, soft):
    rows = []
    for L in LABELS:
        g, p = gold[L].astype(int), pred[L].astype(int)
        tp = int(((g == 1) & (p == 1)).sum()); fp = int(((g == 0) & (p == 1)).sum())
        fn = int(((g == 1) & (p == 0)).sum()); tn = int(((g == 0) & (p == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
        kappa = cohen_kappa_score(g, p) if p.nunique() > 1 or g.nunique() > 1 else np.nan
        auc = roc_auc_score(g, soft[L]) if g.nunique() > 1 else np.nan
        rows.append(dict(label=L, pos_gold=int(g.sum()), pos_weak=int(p.sum()), TP=tp, FP=fp,
                         FN=fn, TN=tn, precision=prec, recall=rec, F1=f1,
                         accuracy=(tp + tn) / len(g), kappa=kappa, AUC_soft=auc))
    return pd.DataFrame(rows).set_index("label")


def bootstrap_ci(gold, pred, soft, n_boot=1000, seed=SEED):
    """IC 95 % por bootstrap: se remuestrean los ESTUDIOS (filas) con reemplazo y
    se recalculan las métricas; el IC son los percentiles 2.5 y 97.5."""
    rng = np.random.default_rng(seed)
    n = len(gold)
    metrics = ["precision", "recall", "F1", "kappa", "AUC_soft"]
    per = {m: [] for m in metrics}
    glob = {"exactitud": [], "F1_macro": [], "kappa_macro": [], "AUC_macro": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        gb, pb, sb = (x.iloc[idx].reset_index(drop=True) for x in (gold, pred, soft))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = per_label(gb, pb, sb)
        for m in metrics:
            per[m].append(t[m].values)
        glob["exactitud"].append((gb.values == pb.values).mean())
        glob["F1_macro"].append(t.F1.mean())
        glob["kappa_macro"].append(np.nanmean(t.kappa))
        glob["AUC_macro"].append(np.nanmean(t.AUC_soft))
    ci = pd.DataFrame(index=LABELS)
    for m in metrics:
        arr = np.array(per[m], dtype=float)
        ci[f"{m}_lo"] = np.nanpercentile(arr, 2.5, axis=0)
        ci[f"{m}_hi"] = np.nanpercentile(arr, 97.5, axis=0)
    gci = {k: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for k, v in glob.items()}
    return ci, gci


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(META / "train.csv")
    weak = pd.read_csv(META / "weak_labels.csv")
    lang = pd.read_csv(META / "study_catalog.csv")[["StudyInstanceUID", "lang"]]

    gold = train[train[LABELS].notna().any(axis=1)][["StudyInstanceUID"] + LABELS]
    df = gold.merge(weak, on="StudyInstanceUID", suffixes=("_gold", "")).merge(lang)
    g = df[[f"{L}_gold" for L in LABELS]].set_axis(LABELS, axis=1)
    soft = df[[f"{L}_p" for L in LABELS]].set_axis(LABELS, axis=1)
    u_ones = df[LABELS]
    u_zeros = u_ones.copy()
    for L in LABELS:
        u_zeros.loc[df[f"{L}_status"] == "uncertain", L] = 0

    pd.set_option("display.width", 200)
    results = {}
    for name, pred in (("U-ones", u_ones), ("U-zeros", u_zeros)):
        t = per_label(g, pred, soft)
        ci, gci = bootstrap_ci(g, pred, soft)
        t = t.join(ci)
        results[name] = t
        t.round(3).to_csv(OUT / f"weak_vs_gold_{name}.csv")
        acc = (g.values == pred.values).mean()
        tp = t.TP.sum(); fp = t.FP.sum(); fn = t.FN.sum()
        micro_f1 = 2 * tp / (2 * tp + fp + fn)
        fmt = lambda k: f"[{gci[k][0]:.2f}-{gci[k][1]:.2f}]"  # noqa: E731
        print(f"\n=== {name}: exactitud global {acc:.3f} {fmt('exactitud')} | "
              f"tasa de ruido {1 - acc:.3f} | F1 micro {micro_f1:.3f} | "
              f"F1 macro {t.F1.mean():.3f} {fmt('F1_macro')} | "
              f"kappa macro {t.kappa.mean():.3f} {fmt('kappa_macro')} | "
              f"AUC macro (blando) {t.AUC_soft.mean():.3f} {fmt('AUC_macro')}")
        show = t[["pos_gold", "pos_weak", "TP", "FP", "FN"]].copy()
        for m in ["precision", "recall", "F1", "kappa"]:
            show[m] = [f"{v:.2f} [{lo:.2f}-{hi:.2f}]" for v, lo, hi in
                       zip(t[m], t[f"{m}_lo"], t[f"{m}_hi"])]
        print(show.to_string())

    # Calibración: exactitud de la etiqueta débil según el tipo de evidencia
    long = []
    for L in LABELS:
        long.append(pd.DataFrame({"status": df[f"{L}_status"], "conf": df[f"{L}_conf"],
                                  "correct": (df[L] == g[L]).astype(int)}))
    long = pd.concat(long)
    cal = long.groupby("status").agg(n=("correct", "size"), conf_asignada=("conf", "mean"),
                                     exactitud_real=("correct", "mean")).round(3)
    cal.to_csv(OUT / "weak_confidence_calibration.csv")
    print("\n=== Calibración de la confianza (U-ones)\n" + cal.to_string())

    # Desglose por idioma
    rows = []
    for lg, idx in df.groupby("lang").groups.items():
        gi, pi = g.loc[idx], u_ones.loc[idx]
        tp = ((gi == 1) & (pi == 1)).values.sum(); fp = ((gi == 0) & (pi == 1)).values.sum()
        fn = ((gi == 1) & (pi == 0)).values.sum()
        rows.append(dict(lang=lg, estudios=len(idx), exactitud=(gi.values == pi.values).mean(),
                         precision=tp / (tp + fp) if tp + fp else np.nan,
                         recall=tp / (tp + fn) if tp + fn else np.nan,
                         F1_micro=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan))
    by_lang = pd.DataFrame(rows).set_index("lang").sort_values("estudios", ascending=False)
    by_lang.round(3).to_csv(OUT / "weak_vs_gold_by_language.csv")
    print("\n=== Por idioma (U-ones; ojo: n muy pequeño)\n" + by_lang.round(3).to_string())


if __name__ == "__main__":
    main()
