"""Paso 9: interpretabilidad con Grad-CAM (Selvaraju et al., 2017) — entregable 5.

Para una corrida entrenada (best.pt + config.json + preds_gold_val.csv de train.py):
  1. Casos: por hallazgo, los N dorados POSITIVOS con mayor probabilidad (aciertos) y el
     dorado NEGATIVO con mayor probabilidad (falso positivo), para mostrar también errores.
  2. Corte: el de mayor atención del modelo para ese hallazgo (la cabeza MIL elige
     dónde mirar dentro de la pila).
  3. Mapa: se pasa el estudio completo (la atención participa del gradiente) y en ese corte
         mapa = ReLU( Σ_k α_k A^k ),  α_k = media espacial de ∂logit_hallazgo / ∂A^k
     con A = salida de layer4 en B (7×7) o del último bloque conv de A, antes del
     max-pool (14×14). Se interpola a 224×224 y se superpone al corte.
  4. Atención a nivel de corte (cuantitativo, todos los dorados): posición relativa del
     corte más atendido. Si el modelo es anatómicamente coherente, los meniscos quedan en
     los extremos de la pila sagital (medial y lateral en extremos OPUESTOS: cuál es cuál
     depende de si la rodilla es izquierda o derecha) y el LCA cerca del centro.

Salidas en reports/gradcam/<corrida>/: gradcam_<hallazgo>.png, gradcam_panel.png,
attention_position.png, cases.csv y attention_summary.txt.
Las imágenes derivan de Kaggle: no se suben al repositorio (.gitignore).

Uso: python gradcam.py --run <carpeta de la corrida> [--labels "Medial Meniscus" ACL Effusion]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common import LABELS, META, ROOT  # noqa: E402
from models import KneeMIL  # noqa: E402

PROC = ROOT / "data" / "processed"
CROP = 224
DEFAULT_LABELS = ["Medial Meniscus", "Lateral Meniscus", "ACL", "Effusion", "Contusion"]
LABEL_ES = {"ACL": "LCA", "MCL": "LCM", "Medial Meniscus": "Menisco medial",
            "Lateral Meniscus": "Menisco lateral", "Medial OA": "Artrosis medial",
            "Lateral OA": "Artrosis lateral", "PF OA": "Artrosis patelofemoral",
            "Effusion": "Derrame", "Synovitis": "Sinovitis", "Baker's": "Quiste de Baker",
            "Contusion": "Contusión", "Fracture": "Fractura"}


def load_model(run, device):
    cfg = json.loads((run / "config.json").read_text())
    model = KneeMIL(cfg["arch"], cfg["pool"], pretrained=False).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.eval()
    if cfg["arch"] == "resnet50":
        target = model.encoder.net.layer4
    else:   # último bloque conv-BN-ReLU ×2 + MaxPool: se toma la ReLU previa al pool
        target = model.encoder.features[-1][5]
    return model, target, cfg


def load_volumes(view, uids):
    index = pd.read_csv(PROC / f"{view}_index.csv").set_index("StudyInstanceUID")
    vols = {}
    for shard, g in index.loc[list(uids)].groupby("shard"):
        arr = np.load(PROC / view / shard)
        for uid in g.index:
            v = arr[uid]
            o = (v.shape[1] - CROP) // 2
            vols[uid] = v[:, o:o + CROP, o:o + CROP]
    return vols


def grad_cam(model, target, vol, label_idx, device):
    """Devuelve (mapa 224×224 en [0,1] del corte más atendido, índice del corte, prob, atención)."""
    store = {}

    def fwd_hook(_, __, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = target.register_forward_hook(fwd_hook)
    x = torch.from_numpy(vol.astype(np.float32) / 255)[None].to(device)
    with torch.enable_grad():
        logits, att = model(x)
        model.zero_grad()
        logits[0, label_idx].backward()
    h.remove()
    att = att[0, :, label_idx].detach().cpu().numpy()
    s = int(att.argmax())
    A, G = store["act"][s].detach(), store["grad"][s]          # (C, h, w)
    cam = F.relu((G.mean((1, 2))[:, None, None] * A).sum(0))
    cam = F.interpolate(cam[None, None], size=(CROP, CROP), mode="bilinear", align_corners=False)[0, 0]
    cam = cam.cpu().numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam, s, float(torch.sigmoid(logits[0, label_idx].detach())), att


def pick_cases(preds, gold, L, n_pos):
    d = preds[["StudyInstanceUID", f"{L}_prob"]].merge(gold[["StudyInstanceUID", f"{L}_gold"]],
                                                        on="StudyInstanceUID")
    d = d.sort_values(f"{L}_prob", ascending=False)
    pos = d[d[f"{L}_gold"] == 1].head(n_pos).assign(kind="acierto (positivo dorado)")
    neg = d[d[f"{L}_gold"] == 0].head(1).assign(kind="falso positivo")
    return pd.concat([pos, neg])


def show(ax, img, cam, title):
    # mezcla RGB explícita: alpha como arreglo no se respeta en todas las versiones de matplotlib
    rgb = np.repeat(img[..., None] / 255.0, 3, axis=2)
    if cam is not None:
        heat = plt.get_cmap("jet")(cam)[..., :3]
        a = (0.55 * cam * (cam > 0.2))[..., None]
        rgb = rgb * (1 - a) + heat * a
    ax.imshow(rgb)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def attention_positions(preds, depth, labels):
    rows = []
    for L in labels:
        pos = preds[f"{L}_slice"].values / np.maximum(depth - 1, 1)
        rows.append(dict(label=L, mediana_pos=np.median(pos),
                         mediana_dist_centro=np.median(np.abs(pos - 0.5)),
                         en_tercio_central=np.mean(np.abs(pos - 0.5) < 1 / 6)))
    out = pd.DataFrame(rows)
    pm = preds["Medial Meniscus_slice"].values / np.maximum(depth - 1, 1)
    pl = preds["Lateral Meniscus_slice"].values / np.maximum(depth - 1, 1)
    opposite = np.mean((pm - 0.5) * (pl - 0.5) < 0)
    return out, opposite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="carpeta con best.pt, config.json y preds_gold_val.csv")
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    ap.add_argument("--n", type=int, default=3, help="aciertos por hallazgo (más 1 falso positivo)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    run = Path(a.run)
    out = Path(a.out) if a.out else ROOT / "reports" / "gradcam" / run.name
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, target, cfg = load_model(run, device)
    preds = pd.read_csv(run / "preds_gold_val.csv")
    gold = pd.read_csv(META / "sample_split.csv")
    gold = gold[gold.StudyInstanceUID.isin(preds.StudyInstanceUID)]

    cases = pd.concat([pick_cases(preds, gold, L, a.n).assign(label=L) for L in a.labels])
    vols = load_volumes(cfg["view"], cases.StudyInstanceUID.unique())
    rows, panel = [], []
    for L in a.labels:
        c = cases[cases.label == L]
        if c.empty:
            continue
        fig, axes = plt.subplots(2, len(c), figsize=(2.6 * len(c), 5.6), squeeze=False)
        for j, r in enumerate(c.itertuples()):
            vol = vols[r.StudyInstanceUID]
            cam, s, p, att = grad_cam(model, target, vol, LABELS.index(L), device)
            tag = "TP" if r.kind.startswith("acierto") else "FP"
            show(axes[0, j], vol[s], None, f"{tag} · corte {s + 1}/{len(vol)}")
            show(axes[1, j], vol[s], cam, f"p = {p:.2f}")
            rows.append(dict(label=L, kind=r.kind, StudyInstanceUID=r.StudyInstanceUID, prob=p,
                             slice=s, depth=len(vol), att_max=float(att.max()),
                             cam_row=float(np.average(np.arange(CROP), weights=cam.sum(1) + 1e-9)),
                             cam_col=float(np.average(np.arange(CROP), weights=cam.sum(0) + 1e-9))))
            if j == 0:
                panel.append((L, vol[s], cam, p, s, len(vol)))
        fig.suptitle(f"Grad-CAM · {LABEL_ES.get(L, L)} · {run.name}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"gradcam_{L.replace(' ', '_').replace(chr(39), '')}.png", dpi=150)
        plt.close(fig)

    # panel resumen: el mejor acierto de cada hallazgo (figura para el informe)
    fig, axes = plt.subplots(2, len(panel), figsize=(2.6 * len(panel), 5.6), squeeze=False)
    for j, (L, img, cam, p, s, D) in enumerate(panel):
        show(axes[0, j], img, None, f"{LABEL_ES.get(L, L)}\ncorte {s + 1}/{D}")
        show(axes[1, j], img, cam, f"p = {p:.2f}")
    fig.tight_layout()
    fig.savefig(out / "gradcam_panel.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "cases.csv", index=False)

    # atención a nivel de corte sobre todos los dorados
    index = pd.read_csv(PROC / f"{cfg['view']}_index.csv").set_index("StudyInstanceUID")
    depth = index.loc[preds.StudyInstanceUID, "n_kept"].values
    summ, opposite = attention_positions(preds, depth, LABELS)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    data = [preds[f"{L}_slice"].values / np.maximum(depth - 1, 1) for L in LABELS]
    ax.boxplot(data, vert=False, tick_labels=[LABEL_ES[L] for L in LABELS])
    ax.axvline(0.5, ls="--", c="gray", lw=0.8)
    ax.set_xlabel("posición relativa del corte más atendido (0 = un extremo, 1 = el otro)")
    fig.tight_layout()
    fig.savefig(out / "attention_position.png", dpi=150)
    plt.close(fig)
    txt = (f"Atención a nivel de corte en gold_val ({len(preds)} estudios, {run.name})\n"
           f"Meniscos medial y lateral atendidos en extremos opuestos de la pila: {opposite:.0%}\n\n"
           + summ.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    (out / "attention_summary.txt").write_text(txt, encoding="utf8")
    print(txt)
    print(f"\nFiguras en {out}")


if __name__ == "__main__":
    main()
