"""Paso 4: muestreo estratificado de estudios y particiones sin fuga.

Población: estudios NO dorados con serie sagital FS. Se eligen N_SAMPLE estudios:
  1. Cupos por hallazgo: para cada etiqueta (de la más rara a la más común) se
     añaden estudios con positivo de ALTA confianza (pos_concl / pos_body) hasta
     tener al menos MIN_POS. Así fractura y contusión quedan representadas sin
     llenar el cupo con menciones inciertas.
  2. Cupo por idioma: al menos MIN_LANG estudios de cada idioma (entregable 7).
  3. Relleno aleatorio hasta N_SAMPLE (mantiene prevalencias cercanas a las reales).
Luego se divide en weak_train / weak_val (85/15) eligiendo, entre N_TRIES semillas,
la partición con prevalencias por etiqueta e idioma más parecidas.
Los 58 dorados forman gold_val y nunca entran a entrenamiento.

Etiqueta dura de entrenamiento: U-zeros (incierta = 0). Se conservan <L>_p
(objetivo blando) y <L>_conf para la ablación de manejo de ruido.
"""
import numpy as np
import pandas as pd

from common import LABELS, META, ROOT, SEED

N_SAMPLE, VAL_FRAC = 600, 0.15
MIN_POS, MIN_LANG, N_TRIES = 60, 15, 300
HIGH_CONF = {"pos_concl", "pos_body"}
POS = {"pos_concl", "pos_body", "pos_weak"}


def load():
    st = pd.read_csv(META / "study_series.csv")
    weak = pd.read_csv(META / "weak_labels.csv")
    train = pd.read_csv(META / "train.csv")
    gold = train[["StudyInstanceUID"] + LABELS].rename(columns={L: f"{L}_gold" for L in LABELS})
    df = st.merge(weak, on="StudyInstanceUID").merge(gold, on="StudyInstanceUID")
    for L in LABELS:
        df[L] = df[f"{L}_status"].isin(POS).astype(int)             # U-zeros
        df[f"{L}_hc"] = df[f"{L}_status"].isin(HIGH_CONF).astype(int)
    return df


def sample(df, rng):
    pool = df[~df.gold & df.sag_fs_series.notna()].copy()
    pool = pool.iloc[rng.permutation(len(pool))]
    chosen = []

    def take(cands, k):
        cands = [u for u in cands if u not in set(chosen)]
        chosen.extend(cands[:max(k, 0)])

    hc = pool.set_index("StudyInstanceUID")[[f"{L}_hc" for L in LABELS]]
    for L in sorted(LABELS, key=lambda l: hc[f"{l}_hc"].sum()):
        have = hc.loc[chosen, f"{L}_hc"].sum() if chosen else 0
        take(hc.index[hc[f"{L}_hc"] == 1].tolist(), MIN_POS - have)
    for lg, g in pool.groupby("lang"):
        have = pool.set_index("StudyInstanceUID").loc[chosen, "lang"].eq(lg).sum()
        take(g.StudyInstanceUID.tolist(), MIN_LANG - have)
    take(pool.StudyInstanceUID.tolist(), N_SAMPLE - len(chosen))
    return pool.set_index("StudyInstanceUID").loc[chosen].reset_index()


def split(s, rng):
    """Mejor de N_TRIES particiones aleatorias: mínima diferencia de prevalencia."""
    feats = pd.concat([s[LABELS], pd.get_dummies(s.lang).astype(int)], axis=1).values
    target = feats.mean(0)
    n_val = int(round(VAL_FRAC * len(s)))
    best, best_cost = None, np.inf
    for _ in range(N_TRIES):
        idx = rng.permutation(len(s))[:n_val]
        val = np.zeros(len(s), bool); val[idx] = True
        # cada etiqueta debe tener al menos 3 positivos en validación
        if (feats[val][:, :len(LABELS)].sum(0) < 3).any():
            continue
        cost = np.abs(feats[val].mean(0) - target).max()
        if cost < best_cost:
            best, best_cost = val, cost
    return np.where(best, "weak_val", "weak_train"), best_cost


def main():
    rng = np.random.default_rng(SEED)
    df = load()
    s = sample(df, rng)
    s["split"], cost = split(s, rng)
    gold = df[df.gold].assign(split="gold_val")
    out = pd.concat([s, gold], ignore_index=True)

    assert out.StudyInstanceUID.is_unique, "estudio repetido entre particiones"
    assert not out[out.split != "gold_val"].gold.any(), "dorado en entrenamiento"

    keep = ["StudyInstanceUID", "split", "lang", "gold", "sag_fs_series", "sag_fs_slices",
            "sag_fs_is3d", "sag_fs_mb", "cor_fs_series", "cor_fs_slices", "cor_fs_is3d", "cor_fs_mb"]
    lab_cols = [c for L in LABELS for c in (L, f"{L}_p", f"{L}_conf", f"{L}_status", f"{L}_gold")]
    out[keep + lab_cols].to_csv(META / "sample_split.csv", index=False)
    (ROOT / "splits").mkdir(exist_ok=True)
    out[["StudyInstanceUID", "split"]].to_csv(ROOT / "splits" / "sample_split_ids.csv", index=False)

    # Prevalencia por partición (entregable 6). En gold_val se usa la etiqueta dorada.
    rows = {}
    for sp, g in out.groupby("split"):
        cols = [f"{L}_gold" for L in LABELS] if sp == "gold_val" else LABELS
        rows[(sp, "n_pos")] = g[cols].sum().values
        rows[(sp, "prev_%")] = (g[cols].mean().values * 100).round(1)
    prev = pd.DataFrame(rows, index=LABELS)
    tr = out[out.split == "weak_train"]
    prev[("weak_train", "pos_weight")] = ((len(tr) - tr[LABELS].sum()) / tr[LABELS].sum()).round(2).values
    prev.to_csv(ROOT / "reports" / "sample_prevalence.csv")

    print(out.split.value_counts().to_string())
    print(f"\nDesbalance máx. de prevalencia train/val: {cost:.3f}")
    print("\nPrevalencias (U-zeros en weak_*, dorada en gold_val):")
    print(prev.to_string())
    print("\nIdiomas por partición:")
    print(pd.crosstab(out.lang, out.split).to_string())
    print(f"\nSeries 3D (se submuestrean): {int(s.sag_fs_is3d.fillna(False).astype(bool).sum())} | "
          f"MB a leer del zip (sag): {out.sag_fs_mb.sum() / 1e3:.1f} GB | "
          f"(cor): {out.cor_fs_mb.sum() / 1e3:.1f} GB")


if __name__ == "__main__":
    main()
