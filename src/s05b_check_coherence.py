"""Paso 5b: chequeos de coherencia entre muestra (s04) e imágenes extraídas (s05).

- Todos los estudios de la muestra tienen volumen; ninguno dorado en entrenamiento.
- Profundidad, resolución original y fabricantes por partición (sesgo de dominio).
- Intensidad: volúmenes vacíos o saturados.
- Mosaico visual: cortes al 20/50/80 % de la pila para casos positivos de varios
  hallazgos -> reports/qa_montage_<view>.png
"""
import argparse

import cv2
import numpy as np
import pandas as pd

from common import LABELS, META, ROOT

PROC = ROOT / "data" / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", choices=["sag", "cor"], default="sag")
    args = ap.parse_args()

    sample = pd.read_csv(META / "sample_split.csv")
    index = pd.read_csv(PROC / f"{args.view}_index.csv")
    df = sample.merge(index, on="StudyInstanceUID", how="left")
    has_series = df[f"{args.view}_fs_series"].notna()
    # error vacío se lee como NaN; un estudio sin fila en el índice no tiene shard
    ok = df.shard.notna() & (df.error.isna() | (df.error == ""))

    print("=== Cobertura")
    print(pd.crosstab(df.split, np.where(ok, "con volumen", np.where(has_series, "ERROR", "sin serie"))))
    assert not df[(df.split != "gold_val") & df.gold].shape[0], "fuga: dorado en train/val"
    assert set(df.loc[df.split == "weak_train", "StudyInstanceUID"]).isdisjoint(
        df.loc[df.split == "weak_val", "StudyInstanceUID"]), "fuga: train y val comparten estudios"
    print("Sin fuga entre particiones: OK")

    d = df[ok]
    print("\n=== Cortes conservados por partición")
    print(d.groupby("split").n_kept.describe().round(1).to_string())
    print("\n=== Resolución original (filas) por partición")
    print(d.groupby("split").rows.describe().round(0).to_string())
    print("\n=== Fabricante (%) por partición")
    man = d.manufacturer.str.upper().str.extract(r"(GE|SIEMENS|PHILIPS|TOSHIBA|CANON|HITACHI)")[0]
    print((pd.crosstab(man.fillna("OTRO"), d.split, normalize="columns") * 100).round(1).to_string())

    # Intensidad y montaje
    stats, picks = [], {}
    want = {"Medial Meniscus": None, "ACL": None, "Effusion": None, "Baker's": None,
            "Fracture": None, "Contusion": None}
    for shard, g in d.groupby("shard"):
        arr = np.load(PROC / args.view / shard)
        for _, r in g.iterrows():
            v = arr[r.StudyInstanceUID]
            stats.append(dict(mean=v.mean(), zeros=(v == 0).mean(), sat=(v == 255).mean()))
            for L in want:
                if want[L] is None and r.split == "gold_val" and r[f"{L}_gold"] == 1:
                    want[L] = r.StudyInstanceUID
                    picks[r.StudyInstanceUID] = v
    st = pd.DataFrame(stats)
    print("\n=== Intensidad (uint8)")
    print(st.describe().round(3).loc[["mean", "min", "max"]].to_string())
    print(f"Volúmenes casi vacíos (media < 5): {(st['mean'] < 5).sum()}")

    tiles = []
    for L, uid in want.items():
        if uid is None:
            continue
        v = picks[uid]
        row = [v[int(f * (len(v) - 1))] for f in (0.2, 0.5, 0.8)]
        row = [cv2.putText(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), t, (6, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
               for im, t in zip(row, [f"{L} 20%", "50%", "80%"])]
        tiles.append(np.hstack(row))
    out = ROOT / "reports" / f"qa_montage_{args.view}.png"
    cv2.imwrite(str(out), np.vstack(tiles))
    print(f"\nMosaico de control: {out}")


if __name__ == "__main__":
    main()
