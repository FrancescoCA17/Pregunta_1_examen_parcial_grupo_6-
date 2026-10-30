"""Comando único: reproduce la pregunta 1 de extremo a extremo.

    python run_all.py                        # todo: datos -> entrenamiento -> evaluación -> Grad-CAM
    python run_all.py --stages data          # solo la parte local (requiere el zip de Kaggle)
    python run_all.py --stages train,eval,gradcam --runs B_base A_base
    python run_all.py --dry-run              # muestra los comandos sin ejecutarlos

Etapas:
  data     s00–s05b: CSV del zip, catálogo, selección de series, etiquetas débiles y su
           validación contra los 58 dorados, muestreo estratificado, extracción de
           imágenes (sagital y coronal) y chequeos de coherencia.
  pack     s07: colab_bundle.zip / colab_code.zip para entrenar en Colab (opcional).
  train    las corridas de RUNS (modelos A y B, ablaciones y semillas) -> runs/<nombre>/.
  eval     evaluate.py sobre runs/ -> reports/eval_*.
  gradcam  gradcam.py para B_base y A_base -> reports/gradcam/.

Todo es reanudable: la extracción omite shards existentes y cada corrida retoma desde
runs/<nombre>/last.pt. El entrenamiento necesita GPU (en CPU es inviable).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

DATA = [
    ["s00_extract_meta.py"],
    ["s01_catalog.py"],
    ["s02_select_series.py"],
    ["s03_weak_labels.py"],
    ["s03b_validate_weak_labels.py"],
    ["s04_sample_split.py"],
    ["s05_extract_images.py", "--view", "sag"],
    ["s05_extract_images.py", "--view", "cor"],
    ["s05b_check_coherence.py", "--view", "sag"],
    ["s05b_check_coherence.py", "--view", "cor"],
]

# Cada ablación cambia UN factor frente a B_base (ver reports/v2/README.md)
RUNS = {
    "A_base": ["--arch", "scratch"],
    "B_base": ["--arch", "resnet50"],
    "B_focal": ["--arch", "resnet50", "--loss", "focal"],
    "B_cor": ["--arch", "resnet50", "--view", "cor"],
    "B_conf": ["--arch", "resnet50", "--noise", "conf"],
    "B_hc": ["--arch", "resnet50", "--noise", "hc"],
    "B_nopw": ["--arch", "resnet50", "--no-pos-weight"],
    "B_all": ["--arch", "resnet50", "--unfreeze", "all"],
    "B_seed1": ["--arch", "resnet50", "--seed", "1"],
    "B_seed2": ["--arch", "resnet50", "--seed", "2"],
}
STAGES = ["data", "pack", "train", "eval", "gradcam"]


def run(cmd, dry):
    shown = " ".join(c if " " not in c else f'"{c}"' for c in cmd)
    print(f"\n$ python {shown}", flush=True)
    if dry:
        return
    t0 = time.time()
    subprocess.run([sys.executable, "-X", "utf8", *cmd], cwd=SRC, check=True)
    print(f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="data,train,eval,gradcam",
                    help=f"lista separada por comas de {STAGES}")
    ap.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    stages = [s.strip() for s in a.stages.split(",")]
    bad = set(stages) - set(STAGES)
    if bad:
        ap.error(f"etapas desconocidas: {sorted(bad)}")
    runs_dir = Path(a.runs_dir)

    if "train" in stages and not a.dry_run:
        import torch
        if not torch.cuda.is_available():
            print("Aviso: no hay GPU. Entrenar en CPU tomaría días; usa Colab "
                  "(notebooks/train_colab.ipynb) o una máquina con CUDA.", flush=True)

    if "data" in stages:
        for cmd in DATA:
            run(cmd, a.dry_run)
    if "pack" in stages:
        run(["s07_pack_colab.py"], a.dry_run)
    if "train" in stages:
        for name in a.runs:
            run(["train.py", *RUNS[name], "--resume", "--out", str(runs_dir / name)], a.dry_run)
    if "eval" in stages:
        run(["evaluate.py", "--runs", str(runs_dir), "--ref", "B_base"], a.dry_run)
    if "gradcam" in stages:
        for name in ("B_base", "A_base"):
            run(["gradcam.py", "--run", str(runs_dir / name)], a.dry_run)
    print("\nListo.")


if __name__ == "__main__":
    main()
