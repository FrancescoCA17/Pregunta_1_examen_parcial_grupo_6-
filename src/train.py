"""Paso 6: entrenamiento de los modelos A (desde cero) y B (ResNet-50) con etiquetas débiles.

Datos: volúmenes de data/processed/<view> (s05) y etiquetas de data/meta/sample_split.csv (s04).
  - weak_train: entrenamiento con etiquetas débiles (U-zeros).
  - weak_val: selección del mejor epoch (AUC macro contra etiquetas débiles).
  - gold_val: SOLO se predice al final; nunca se usa para elegir nada.

Factores de la ablación (entregables 4 y 6):
  --view   sag | cor                     plano de MRI
  --loss   bce | focal                   focal = BCE · (1 - p_t)^γ, mismo pos_weight
  --noise  hard | conf | hc              etiqueta dura / ponderar por confianza /
                                         solo etiquetas de alta confianza (conf ≥ 0.8)
  --no-pos-weight                        sin mitigación del desbalance (línea base E6)
  --pool   attn | max                    agregación corte -> estudio

Salidas en --out: config.json, history.csv, best.pt, last.pt (para reanudar con --resume),
preds_weak_val.csv y preds_gold_val.csv (probabilidad y corte de mayor atención por hallazgo).

Uso: python train.py --arch resnet50 --out ../runs/B_base
     python train.py --smoke [--epochs 12]   (prueba en CPU con 6 estudios; con más
     épocas la pérdida de entrenamiento debe caer: el flujo aprende)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from common import LABELS, META, ROOT, SEED
from models import KneeMIL, count_params

PROC = ROOT / "data" / "processed"
CROP = 224
HIGH_CONF = 0.8  # pos_concl 0.95, pos_body 0.85, neg 0.95, ausente 0.80


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["scratch", "resnet50"], default="resnet50")
    ap.add_argument("--view", choices=["sag", "cor"], default="sag")
    ap.add_argument("--loss", choices=["bce", "focal"], default="bce")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--noise", choices=["hard", "conf", "hc"], default="hard")
    ap.add_argument("--no-pos-weight", action="store_true")
    ap.add_argument("--pool", choices=["attn", "max"], default="attn")
    ap.add_argument("--epochs", type=int, default=0, help="0 = 60 para A, 20 para B")
    ap.add_argument("--freeze-epochs", type=int, default=3, help="B: épocas solo con la cabeza")
    ap.add_argument("--unfreeze", choices=["layer34", "all"], default="layer34",
                    help="B: bloques que se reentrenan después de las épocas congeladas")
    ap.add_argument("--batch", type=int, default=4, help="estudios por paso")
    ap.add_argument("--lr", type=float, default=1e-3, help="cabeza (y todo A)")
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--patience", type=int, default=0, help="0 = 15 para A (aprende más lento), 8 para B")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto",
                    help="auto = bf16 si la GPU lo soporta, si no fp16; en CPU fp32")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.epochs, a.freeze_epochs, a.batch, a.workers = a.epochs or 2, 1, 2, 0
    a.epochs = a.epochs or (60 if a.arch == "scratch" else 20)
    a.patience = a.patience or (15 if a.arch == "scratch" else 8)
    if not a.out:
        a.out = str(ROOT / "runs" / ("smoke_" + a.arch if a.smoke else
                                    f"{a.arch}_{a.view}_{a.loss}_{a.noise}_{a.pool}"
                                    + ("_nopw" if a.no_pos_weight else "")))
    return a


# ---------------------------------------------------------------- datos
def load_data(view, smoke):
    sample = pd.read_csv(META / "sample_split.csv")
    index = pd.read_csv(PROC / f"{view}_index.csv")
    index = index[index.error.isna() | (index.error == "")]
    df = sample.merge(index[["StudyInstanceUID", "shard", "fallback"]], on="StudyInstanceUID")
    if smoke:
        df = pd.concat([g.head(6 if s == "weak_train" else 3) for s, g in df.groupby("split")])
    vols = {}
    for shard, g in df.groupby("shard"):
        arr = np.load(PROC / view / shard)
        for uid in g.StudyInstanceUID:
            vols[uid] = arr[uid]
    return df.reset_index(drop=True), vols


def label_weights(df, noise):
    conf = df[[f"{L}_conf" for L in LABELS]].values.astype(np.float32)
    if noise == "conf":
        return conf
    if noise == "hc":
        return (conf >= HIGH_CONF).astype(np.float32)
    return np.ones_like(conf)


class Studies(torch.utils.data.Dataset):
    def __init__(self, df, vols, noise="hard"):
        self.uids = df.StudyInstanceUID.tolist()
        self.vols = vols
        self.y = df[LABELS].values.astype(np.float32)          # U-zeros (s04)
        self.w = label_weights(df, noise)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        return torch.from_numpy(self.vols[self.uids[i]]), self.y[i], self.w[i], i


def collate(batch):
    D = max(v.shape[0] for v, *_ in batch)
    H = batch[0][0].shape[1]
    x = torch.zeros(len(batch), D, H, H, dtype=torch.uint8)
    mask = torch.zeros(len(batch), D, dtype=torch.bool)
    for k, (v, *_) in enumerate(batch):
        x[k, :len(v)], mask[k, :len(v)] = v, True
    y = torch.tensor(np.stack([b[1] for b in batch]))
    w = torch.tensor(np.stack([b[2] for b in batch]))
    return x, mask, y, w, torch.tensor([b[3] for b in batch])


def augment(x, mask, train):
    """En GPU. x uint8 (B, D, 256, 256) -> float (B, D, 224, 224) en [0, 1]."""
    x = x.float() / 255
    B, D, H, _ = x.shape
    if not train:
        o = (H - CROP) // 2
        return x[:, :, o:o + CROP, o:o + CROP], mask
    out = torch.empty(B, D, CROP, CROP, device=x.device)
    for k in range(B):
        i, j = np.random.randint(0, H - CROP + 1, 2)
        v = x[k, :, i:i + CROP, j:j + CROP]
        if np.random.rand() < 0.5:   # orientación anterior/posterior variable entre centros
            v = v.flip(-1)
        n = int(mask[k].sum())
        if np.random.rand() < 0.5:   # rodilla izquierda vs derecha: medial/lateral invertidos
            v = torch.cat([v[:n].flip(0), v[n:]])
        g = np.exp(np.random.uniform(-0.2, 0.2))   # gamma y contraste leves
        v = v.clamp(0, 1) ** g * np.random.uniform(0.9, 1.1)
        out[k] = v.clamp(0, 1)
    return out, mask


# ---------------------------------------------------------------- pérdida
def loss_fn(logits, y, w, pos_weight, kind, gamma):
    """BCE o focal por etiqueta, ponderada por w (ruido) y pos_weight (desbalance)."""
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    if kind == "focal":
        p = torch.sigmoid(logits)
        p_t = torch.where(y > 0.5, p, 1 - p)
        bce = bce * (1 - p_t) ** gamma
    return (bce * w).sum() / w.sum().clamp(min=1)


def macro_auc(y, p):
    aucs = [roc_auc_score(y[:, l], p[:, l]) for l in range(y.shape[1]) if 0 < y[:, l].sum() < len(y)]
    return float(np.mean(aucs)) if aucs else float("nan")


# ---------------------------------------------------------------- entrenamiento
def set_train(model, arch, backbone_frozen, unfreeze="layer34"):
    """B: BN preentrenada congelada (el lote son cortes casi iguales de un estudio) y,
    tras las épocas solo-cabeza, se reentrenan layer3/layer4 (o todo con unfreeze="all").
    v1 reentrenaba toda la red con LR 1e-4: la AUC en weak_val cayó de 0,71 a 0,54 al
    descongelar y la pérdida volvió al nivel de un predictor constante."""
    model.train()
    if arch == "resnet50":
        for m in model.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for n, p in model.encoder.named_parameters():
            stage_ok = unfreeze == "all" or n.startswith(("net.layer3", "net.layer4"))
            p.requires_grad = (not backbone_frozen and stage_ok
                               and ".bn" not in n and "downsample.1" not in n)


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    probs, atts, idx = [], [], []
    for x, mask, _, _, i in loader:
        x, mask = augment(x.to(device), mask.to(device), train=False)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            logits, att = model(x, mask)
        probs.append(torch.sigmoid(logits.float()).cpu())
        atts.append(att.float().argmax(1).cpu())          # corte de mayor atención por etiqueta
        idx.append(i)
    return torch.cat(probs).numpy(), torch.cat(atts).numpy(), torch.cat(idx).numpy()


def save_preds(path, df, probs, atts, idx):
    d = df.iloc[idx][["StudyInstanceUID", "split", "lang", "fallback"]].reset_index(drop=True)
    for k, L in enumerate(LABELS):
        d[f"{L}_prob"] = probs[:, k]
        d[f"{L}_slice"] = atts[:, k]
    d.to_csv(path, index=False)


def main(argv=None):
    a = get_args(argv)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}.get(a.precision)
    if a.precision == "auto" and device.type == "cuda":
        amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))

    df, vols = load_data(a.view, a.smoke)
    parts = {s: df[df.split == s].reset_index(drop=True) for s in ("weak_train", "weak_val", "gold_val")}
    ds = {s: Studies(d, vols, a.noise if s == "weak_train" else "hard") for s, d in parts.items()}
    dl = {s: torch.utils.data.DataLoader(d, batch_size=a.batch if s == "weak_train" else 1,
                                         shuffle=s == "weak_train", collate_fn=collate,
                                         num_workers=a.workers, pin_memory=device.type == "cuda")
          for s, d in ds.items()}
    # monitor de ajuste: AUC en un subconjunto fijo de entrenamiento, sin aumentos.
    # Si queda cerca de 0,5 el modelo no está aprendiendo (subajuste), no es ruido de validación.
    mon = parts["weak_train"].sample(min(90, len(parts["weak_train"])), random_state=a.seed)
    ds["train_mon"] = Studies(mon.reset_index(drop=True), vols)
    dl["train_mon"] = torch.utils.data.DataLoader(ds["train_mon"], batch_size=1, collate_fn=collate,
                                                  num_workers=a.workers)
    y_tr = ds["weak_train"].y
    pw = None if a.no_pos_weight else torch.tensor(
        (len(y_tr) - y_tr.sum(0)) / np.maximum(y_tr.sum(0), 1), device=device)
    print(f"Dispositivo {device} ({amp}) | {', '.join(f'{s}={len(d)}' for s, d in ds.items())}", flush=True)

    model = KneeMIL(a.arch, a.pool).to(device)
    print(f"{a.arch}: {count_params(model) / 1e6:.2f} M parámetros", flush=True)
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    groups = [{"params": head, "lr": a.lr}]
    enc = list(model.encoder.parameters())
    groups.append({"params": enc, "lr": a.lr if a.arch == "scratch" else a.lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps = a.epochs * len(dl["weak_train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=amp == torch.float16)

    start, best, history = 0, -np.inf, []
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        print(f"Reanudando desde la época {start}", flush=True)

    for ep in range(start, a.epochs):
        t0 = time.time()
        set_train(model, a.arch, backbone_frozen=a.arch == "resnet50" and ep < a.freeze_epochs,
                  unfreeze=a.unfreeze)
        tot, n = 0.0, 0
        for x, mask, y, w, _ in dl["weak_train"]:
            x, mask = augment(x.to(device, non_blocking=True), mask.to(device), train=True)
            y, w = y.to(device), w.to(device)
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                logits, _ = model(x, mask)
            loss = loss_fn(logits.float(), y, w, pw, a.loss, a.gamma)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"pérdida no finita en la época {ep}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot, n = tot + loss.item() * len(y), n + len(y)

        probs, _, idx = predict(model, dl["weak_val"], device, amp)
        auc = macro_auc(ds["weak_val"].y[idx], probs)
        p_tr, _, i_tr = predict(model, dl["train_mon"], device, amp)
        auc_tr = macro_auc(ds["train_mon"].y[i_tr], p_tr)
        history.append(dict(epoch=ep, train_loss=tot / n, train_auc_weak=auc_tr, val_auc_weak=auc,
                            sec=round(time.time() - t0)))
        print(f"época {ep:2d} | pérdida {tot / n:.4f} | AUC train {auc_tr:.4f} | AUC weak_val {auc:.4f} | "
              f"{time.time() - t0:.0f}s", flush=True)
        score = -np.inf if np.isnan(auc) else auc
        if score > best or not (out / "best.pt").exists():
            best = score
            torch.save(model.state_dict(), out / "best.pt")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=ep, best=best, history=history), out / "last.pt")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        recent = [h["val_auc_weak"] for h in history[-a.patience:]]
        if len(history) > a.patience and max(recent, key=lambda v: -np.inf if np.isnan(v) else v) < best:
            print(f"Parada temprana: {a.patience} épocas sin mejorar", flush=True)
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    for s in ("weak_val", "gold_val"):
        save_preds(out / f"preds_{s}.csv", parts[s], *predict(model, dl[s], device, amp))
    print(f"Mejor AUC weak_val {best:.4f}. Predicciones en {out}", flush=True)


if __name__ == "__main__":
    main()
