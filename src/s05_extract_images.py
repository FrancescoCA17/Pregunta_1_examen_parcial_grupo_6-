"""Paso 5: extracción selectiva de imágenes directamente desde el zip.

Para cada estudio de sample_split.csv se lee SOLO la serie elegida (sagital FS por
defecto, coronal FS con --view cor), sin descomprimir el zip:
  1. Orden de cortes por posición a lo largo de la normal del plano
     (ImagePositionPatient · normal), con InstanceNumber como respaldo.
  2. Intensidad: RescaleSlope/Intercept, inversión si MONOCHROME1, y normalización
     por volumen a percentiles 0.5–99.5 -> uint8.
  3. Cada corte se rellena a cuadrado y se redimensiona a SIZE×SIZE (256: deja
     margen para recorte aleatorio a 224 al entrenar).
  4. Se conservan TODOS los cortes (las estructuras están repartidas a lo ancho de
     la pila: menisco medial en un extremo, LCA al centro, menisco lateral en el
     otro). Solo las series 3D (> MAX_SLICES) se submuestrean a SUB_3D cortes.
Salida: data/processed/<view>/shard_XXX.npz (un array (D, SIZE, SIZE) por estudio)
y data/processed/<view>_index.csv. Se puede reanudar: los shards existentes se omiten.

Uso: python s05_extract_images.py --view sag [--limit 5] [--workers 8]
"""
import argparse
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
import pydicom

from common import META, ROOT, ZIP_PATH

SIZE, MAX_SLICES, SUB_3D, SHARD = 256, 60, 32, 50
OUT = ROOT / "data" / "processed"


def series_files(series_uids):
    """Mapa serie -> lista de rutas dentro del zip (una sola pasada por el índice)."""
    wanted, files = set(series_uids), {}
    with zipfile.ZipFile(ZIP_PATH) as z:
        for name in z.namelist():
            parts = name.split("/")
            if len(parts) == 4 and parts[2] in wanted:
                files.setdefault(parts[2], []).append(name)
    return files


def to_square(img):
    h, w = img.shape
    s = max(h, w)
    pad = np.zeros((s, s), img.dtype)
    pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return cv2.resize(pad, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def slice_key(ds):
    try:
        iop = np.array(ds.ImageOrientationPatient, float)
        normal = np.cross(iop[:3], iop[3:])
        return float(np.dot(np.array(ds.ImagePositionPatient, float), normal))
    except Exception:
        return float(getattr(ds, "InstanceNumber", 0) or 0)


def load_series(names):
    """Lee y preprocesa una serie. Devuelve (volumen uint8, metadatos)."""
    slices, meta = [], {}
    with zipfile.ZipFile(ZIP_PATH) as z:
        for n in names:
            ds = pydicom.dcmread(z.open(n))
            img = ds.pixel_array.astype(np.float32)
            if img.ndim == 3:  # multiframe raro: tomar primer frame
                img = img[0]
            img = img * float(getattr(ds, "RescaleSlope", 1) or 1) + \
                float(getattr(ds, "RescaleIntercept", 0) or 0)
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                img = img.max() - img
            slices.append((slice_key(ds), img))
            if not meta:
                meta = dict(rows=int(ds.Rows), cols=int(ds.Columns),
                            spacing=float(ds.PixelSpacing[0]) if "PixelSpacing" in ds else np.nan,
                            thickness=float(getattr(ds, "SliceThickness", np.nan) or np.nan),
                            manufacturer=str(getattr(ds, "Manufacturer", "")),
                            description=str(getattr(ds, "SeriesDescription", "")),
                            transfer_syntax=str(ds.file_meta.TransferSyntaxUID.name))
    slices.sort(key=lambda t: t[0])
    vol = [to_square(img) for _, img in slices]
    n_orig = len(vol)
    if n_orig > MAX_SLICES:
        vol = [vol[i] for i in np.linspace(0, n_orig - 1, SUB_3D).round().astype(int)]
    vol = np.stack(vol).astype(np.float32)
    lo, hi = np.percentile(vol, [0.5, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    meta.update(n_orig=n_orig, n_kept=len(vol))
    return (vol * 255).round().astype(np.uint8), meta


def work(task):
    uid, series_uid, names = task
    t0 = time.time()
    try:
        vol, meta = load_series(names)
        return uid, vol, dict(meta, series=series_uid, error="", sec=round(time.time() - t0, 1))
    except Exception as e:  # se registra y se sigue con el resto
        return uid, None, dict(series=series_uid, error=f"{type(e).__name__}: {e}"[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", choices=["sag", "cor"], default="sag")
    ap.add_argument("--limit", type=int, default=0, help="solo N estudios (prueba)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    col = f"{args.view}_fs_series"
    sample = pd.read_csv(META / "sample_split.csv")
    sample = sample[sample[col].notna()].sort_values(["split", "StudyInstanceUID"])
    if args.limit:
        sample = sample.head(args.limit)
    out_dir = OUT / (args.view if not args.limit else f"{args.view}_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    files = series_files(sample[col])
    print(f"Índice del zip: {time.time() - t0:.0f}s, {len(files)} series localizadas", flush=True)

    index_rows = []
    chunks = [sample.iloc[i:i + SHARD] for i in range(0, len(sample), SHARD)]
    with ProcessPoolExecutor(args.workers) as pool:
        for k, chunk in enumerate(chunks):
            shard = out_dir / f"shard_{k:03d}.npz"
            if shard.exists():
                prev = out_dir / f"shard_{k:03d}.csv"
                if prev.exists():
                    index_rows.append(pd.read_csv(prev))
                print(f"[{k + 1}/{len(chunks)}] {shard.name} ya existe, se omite", flush=True)
                continue
            tasks = [(r.StudyInstanceUID, r[col], sorted(files[r[col]]))
                     for _, r in chunk.iterrows()]
            arrays, rows = {}, []
            for uid, vol, meta in pool.map(work, tasks):
                if vol is not None:
                    arrays[uid] = vol
                rows.append(dict(StudyInstanceUID=uid, shard=shard.name, view=args.view, **meta))
            np.savez_compressed(shard, **arrays)
            part = pd.DataFrame(rows)
            part.to_csv(out_dir / f"shard_{k:03d}.csv", index=False)
            index_rows.append(part)
            n_err = (part.error.fillna("") != "").sum()
            print(f"[{k + 1}/{len(chunks)}] {shard.name}: {len(arrays)} estudios, {n_err} errores, "
                  f"{(time.time() - t0) / 60:.1f} min acumulados", flush=True)

    index = pd.concat(index_rows, ignore_index=True)
    index.to_csv(OUT / f"{out_dir.name}_index.csv", index=False)
    ok = index.error.fillna("") == ""
    print(f"\nListo: {ok.sum()}/{len(index)} estudios en {(time.time() - t0) / 60:.1f} min. "
          f"Tamaño: {sum(f.stat().st_size for f in out_dir.glob('*.npz')) / 1e6:.0f} MB")
    if (~ok).any():
        print(index.loc[~ok, ["StudyInstanceUID", "error"]].to_string())


if __name__ == "__main__":
    main()
