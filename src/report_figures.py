"""Figuras del informe a partir de los historiales de entrenamiento (sin imágenes de Kaggle).

report/figures/curves.pdf:
  (a) Modelo B: AUC en weak_val de v1 (toda la ResNet, LR 1e-4) frente a v2
      (layer3/4, LR 2e-5); la línea punteada marca el descongelamiento.
  (b) AUC en entrenamiento (subconjunto fijo) y en weak_val de A y B v2: B memoriza
      el ruido después de su mejor época, A no llega a ajustar ni el entrenamiento.
Además copia el panel Grad-CAM de B (si existe) a report/figures/gradcam_B.png; esa
imagen deriva de Kaggle y no se versiona.
"""
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from common import ROOT  # noqa: E402

V2 = ROOT / "reports" / "v2"
FIG = ROOT / "report" / "figures"
# Historial de B v1 (reports/v1/README.md): épocas 0-7, descongelado en la época 2
B_V1_VAL = [0.646, 0.710, 0.540, 0.620, 0.558, 0.568, 0.640, 0.654]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    b = pd.read_csv(V2 / "B_base_history.csv")
    a = pd.read_csv(V2 / "A_base_history.csv")
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.3))

    ax1.plot(range(len(B_V1_VAL)), B_V1_VAL, "o-", ms=3, c="tab:red", label="v1: toda la red, LR 1e-4")
    ax1.plot(b.epoch, b.val_auc_weak, "o-", ms=3, c="tab:blue", label="v2: layer3/4, LR 2e-5")
    ax1.axvline(1.5, ls=":", c="tab:red", lw=1)
    ax1.axvline(2.5, ls=":", c="tab:blue", lw=1)
    ax1.set(xlabel="época", ylabel="AUC macro en weak_val", title="(a) Fine-tuning del modelo B", ylim=(0.5, 0.78))
    ax1.legend(frameon=False, loc="lower right", fontsize=6.5)

    ax2.plot(b.epoch, b.train_auc_weak, "-", c="tab:blue", label="B entrenamiento")
    ax2.plot(b.epoch, b.val_auc_weak, "--", c="tab:blue", label="B weak_val")
    ax2.plot(a.epoch, a.train_auc_weak, "-", c="tab:green", label="A entrenamiento")
    ax2.plot(a.epoch, a.val_auc_weak, "--", c="tab:green", label="A weak_val")
    best = b.val_auc_weak.idxmax()
    ax2.axvline(best, ls=":", c="gray", lw=1)
    ax2.annotate("mejor época de B", (best, 0.52), fontsize=7, color="gray", ha="left", xytext=(best + 1, 0.52))
    ax2.set(xlabel="época", ylabel="AUC macro", title="(b) Ajuste frente a generalización", ylim=(0.5, 1.0))
    ax2.legend(frameon=False, loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "curves.pdf")
    print(FIG / "curves.pdf")

    panel = ROOT / "reports" / "gradcam" / "B_base" / "gradcam_panel.png"
    if panel.exists():
        shutil.copy(panel, FIG / "gradcam_B.png")
        print(FIG / "gradcam_B.png")


if __name__ == "__main__":
    main()
