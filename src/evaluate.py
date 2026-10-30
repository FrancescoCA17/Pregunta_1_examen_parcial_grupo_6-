"""Paso 8: evaluación de todas las corridas sobre los 58 dorados (entregables 3, 4 y 6).

Por corrida (carpeta con preds_weak_val.csv y preds_gold_val.csv, salida de train.py):
  1. AUC-ROC macro (métrica oficial) y por etiqueta en gold_val, con IC 95 % por
     bootstrap de estudios (1000 remuestreos). También sin los 2 dorados de respaldo
     (columna fallback), para mostrar que no cambian la conclusión.
  2. Umbral por etiqueta elegido en weak_val (máximo F1 contra la etiqueta débil):
     los dorados no se tocan para elegirlo. Con ese umbral: precisión, recall y F1 en gold.
  3. Diferencia de AUC macro frente a la corrida de referencia (--ref), con IC por
     bootstrap pareado (mismos remuestreos para ambas): si el IC cruza 0, la
     diferencia no es concluyente con 58 estudios.
Filas de referencia: "texto_s03" (AUC del objetivo blando de las etiquetas débiles,
leyendo el reporte) y "prevalencia" (predice lo mismo para todos: AUC 0,5).

Salidas en reports/: eval_summary.csv, eval_per_label.csv, eval_summary.txt
Uso: python evaluate.py [--runs ../runs] [--ref B_base]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import precision_recall_curve

from common import LABELS, META, ROOT, SEED

OUT = ROOT / "reports"
N_BOOT = 1000


def auc(y, p):
    """AUC por rangos (empates promediados); NaN si falta una clase."""
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return np.nan
    r = rankdata(p)
    return (r[y.astype(bool)].sum() - pos * (pos + 1) / 2) / (pos * neg)


def per_label_auc(Y, P, idx=None):
    if idx is not None:
        Y, P = Y[idx], P[idx]
    return np.array([auc(Y[:, k], P[:, k]) for k in range(Y.shape[1])])


def best_threshold(y, p):
    """Umbral que maximiza F1 en weak_val; 0.5 si no hay positivos."""
    if y.sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[np.argmax(f1[:-1])])


def prf(y, pred):
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def load_runs(runs_dir, gold, weak_val):
    runs = {}
    for d in sorted(Path(runs_dir).iterdir()):
        fg, fv = d / "preds_gold_val.csv", d / "preds_weak_val.csv"
        if not (fg.exists() and fv.exists()):
            continue
        g = gold[["StudyInstanceUID"]].merge(pd.read_csv(fg), on="StudyInstanceUID", how="left")
        v = weak_val[["StudyInstanceUID"]].merge(pd.read_csv(fv), on="StudyInstanceUID", how="left")
        cols = [f"{L}_prob" for L in LABELS]
        if g[cols].isna().any().any():
            print(f"[aviso] {d.name}: faltan predicciones de dorados, se omite")
            continue
        runs[d.name] = dict(gold=g[cols].values, val=v[cols].values,
                            val_ok=~v[cols].isna().any(axis=1).values)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--ref", default="B_base")
    a = ap.parse_args()

    s = pd.read_csv(META / "sample_split.csv")
    idx_file = ROOT / "data" / "processed" / "sag_index.csv"
    gold = s[s.split == "gold_val"].reset_index(drop=True)
    if idx_file.exists():
        fb = pd.read_csv(idx_file)[["StudyInstanceUID", "fallback"]]
        gold = gold.merge(fb, on="StudyInstanceUID", how="left").fillna({"fallback": False})
    else:
        gold["fallback"] = False
    weak_val = s[s.split == "weak_val"].reset_index(drop=True)
    Yg = gold[[f"{L}_gold" for L in LABELS]].values.astype(int)
    Yv = weak_val[LABELS].values.astype(int)
    keep56 = ~gold.fallback.astype(bool).values

    runs = load_runs(a.runs, gold, weak_val)
    if not runs:
        raise SystemExit(f"No hay corridas con predicciones en {a.runs}")
    # referencias sin entrenamiento
    preds = {name: r["gold"] for name, r in runs.items()}
    preds["texto_s03"] = gold[[f"{L}_p" for L in LABELS]].values
    preds["prevalencia"] = np.tile(Yv.mean(0), (len(gold), 1))

    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, len(gold), len(gold)) for _ in range(N_BOOT)]
    boot_auc = {n: np.array([per_label_auc(Yg, P, b) for b in boots]) for n, P in preds.items()}

    summary, per_label = [], []
    for name, P in preds.items():
        lab = per_label_auc(Yg, P)
        lo, hi = np.nanpercentile(boot_auc[name], [2.5, 97.5], axis=0)
        macro_b = np.nanmean(boot_auc[name], axis=1)
        row = dict(run=name, auc_macro=np.nanmean(lab),
                   auc_lo=np.percentile(macro_b, 2.5), auc_hi=np.percentile(macro_b, 97.5),
                   auc_macro_56=np.nanmean(per_label_auc(Yg[keep56], P[keep56])))
        if a.ref in boot_auc and name != a.ref:
            d = macro_b - np.nanmean(boot_auc[a.ref], axis=1)
            row.update(delta_vs_ref=row["auc_macro"] - np.nanmean(per_label_auc(Yg, preds[a.ref])),
                       delta_lo=np.percentile(d, 2.5), delta_hi=np.percentile(d, 97.5))
        f1s = []
        for k, L in enumerate(LABELS):
            r = dict(run=name, label=L, gold_pos=int(Yg[:, k].sum()), auc=lab[k], auc_lo=lo[k], auc_hi=hi[k])
            if name in runs:
                ok = runs[name]["val_ok"]
                thr = best_threshold(Yv[ok, k], runs[name]["val"][ok, k])
                pr, rc, f1 = prf(Yg[:, k], P[:, k] >= thr)
                r.update(threshold=thr, precision=pr, recall=rc, f1=f1)
                f1s.append((pr, rc, f1))
            per_label.append(r)
        if f1s:
            row.update(dict(zip(["precision_macro", "recall_macro", "f1_macro"], np.mean(f1s, 0))))
        summary.append(row)

    summ = pd.DataFrame(summary)
    pl = pd.DataFrame(per_label)
    OUT.mkdir(exist_ok=True)
    summ.to_csv(OUT / "eval_summary.csv", index=False)
    pl.to_csv(OUT / "eval_per_label.csv", index=False)

    fmt = lambda x: f"{x:.3f}"  # noqa: E731
    lines = [f"Evaluación en gold_val (n={len(gold)}; {int((~keep56).sum())} con serie de respaldo). "
             f"IC 95 % por bootstrap de estudios ({N_BOOT}).",
             "Umbral por etiqueta: máximo F1 en weak_val contra la etiqueta débil.", "",
             "=== AUC macro y P/R/F1 macro al umbral"]
    cols = [c for c in ["run", "auc_macro", "auc_lo", "auc_hi", "auc_macro_56", "delta_vs_ref", "delta_lo",
                        "delta_hi", "precision_macro", "recall_macro", "f1_macro"] if c in summ]
    lines.append(summ[cols].to_string(index=False, float_format=fmt))
    lines += ["", "=== AUC por etiqueta (filas: hallazgo; columnas: corrida)"]
    lines.append(pl.pivot(index="label", columns="run", values="auc").loc[LABELS].to_string(float_format=fmt))
    if "recall" in pl:
        prev_tr = s.loc[s.split == "weak_train", LABELS].mean()
        rare = list(prev_tr.sort_values().index[:4])
        lines += ["", f"=== Hallazgos más raros en entrenamiento ({', '.join(rare)}): recall al umbral (E6)"]
        lines.append(pl[pl.label.isin(rare) & pl.recall.notna()]
                     .pivot(index="label", columns="run", values="recall").to_string(float_format=fmt))
    txt = "\n".join(lines)
    (OUT / "eval_summary.txt").write_text(txt, encoding="utf8")
    print(txt)


if __name__ == "__main__":
    main()
