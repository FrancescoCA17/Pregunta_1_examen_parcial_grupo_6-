"""Modelos 2.5D con agregación corte -> estudio (aprendizaje de múltiples instancias).

Entrada: un estudio = volumen (D, H, W) uint8. Cada corte se convierte en una imagen
de 3 canales con sus vecinos (i-1, i, i+1): así un codificador 2D ve algo de
contexto 3D y puede usar pesos de ImageNet (que esperan 3 canales).

  codificador 2D por corte -> h (D, F) -> agregación por hallazgo -> 12 logits

Agregación por atención por etiqueta (Ilse et al., 2018, con una cabeza por
hallazgo): cada etiqueta aprende en qué cortes mirar (menisco medial en un extremo
de la pila sagital, LCA al centro, menisco lateral en el otro). Es invariante al
orden y al número de cortes, y los pesos de atención dicen qué corte usar en Grad-CAM.
pool="max" queda como alternativa para la ablación.

Modelo A: CNN desde cero, pequeña (≈1,5 M parámetros) porque hay solo ~510 estudios.
Modelo B: ResNet-50 preentrenada en ImageNet (He et al., 2016), F = 2048.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import LABELS

N_LABELS = len(LABELS)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_25d(vol):
    """(B, D, H, W) en [0, 1] -> (B, D, 3, H, W) con cortes vecinos como canales."""
    prev = torch.cat([vol[:, :1], vol[:, :-1]], 1)
    nxt = torch.cat([vol[:, 1:], vol[:, -1:]], 1)
    x = torch.stack([prev, vol, nxt], 2)
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    return (x - mean) / std


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class ScratchEncoder(nn.Module):
    """Modelo A: 5 bloques VGG-like 224 -> 7, luego promedio global."""

    def __init__(self, widths=(24, 48, 96, 160, 256)):
        super().__init__()
        layers, cin = [], 3
        for w in widths:
            layers.append(conv_block(cin, w))
            cin = w
        self.features = nn.Sequential(*layers)
        self.out_dim = cin

    def forward(self, x):
        return F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)


class ResNetEncoder(nn.Module):
    """Modelo B: ResNet-50 sin la capa final; layer4 es la capa objetivo de Grad-CAM."""

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        net.fc = nn.Identity()
        self.net = net
        self.out_dim = 2048

    def stages(self):
        """Grupos para el fine-tuning por etapas: [tallo+layer1, layer2, layer3, layer4]."""
        n = self.net
        return [nn.ModuleList([n.conv1, n.bn1, n.layer1]), n.layer2, n.layer3, n.layer4]

    def forward(self, x):
        return self.net(x)


class LabelAttentionPool(nn.Module):
    """Atención con compuerta (Ilse et al., 2018), una distribución sobre cortes por etiqueta."""

    def __init__(self, dim, hidden=128, n_labels=N_LABELS):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, n_labels)
        self.cls_w = nn.Parameter(torch.zeros(n_labels, dim))
        self.cls_b = nn.Parameter(torch.zeros(n_labels))
        nn.init.normal_(self.cls_w, std=0.01)

    def forward(self, h, mask):
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))  # (B, D, L)
        a = a.masked_fill(~mask[..., None], float("-inf")).softmax(1)
        z = torch.einsum("bdl,bdf->blf", a, h)                          # (B, L, F)
        return (z * self.cls_w).sum(-1) + self.cls_b, a


class MaxPool(nn.Module):
    def __init__(self, dim, n_labels=N_LABELS):
        super().__init__()
        self.fc = nn.Linear(dim, n_labels)

    def forward(self, h, mask):
        logits = self.fc(h).masked_fill(~mask[..., None], float("-inf"))  # (B, D, L)
        return logits.max(1).values, logits.softmax(1)


class KneeMIL(nn.Module):
    def __init__(self, arch="resnet50", pool="attn", pretrained=True, dropout=0.3):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained) if arch == "resnet50" else ScratchEncoder()
        dim = self.encoder.out_dim
        self.drop = nn.Dropout(dropout)
        self.pool = LabelAttentionPool(dim) if pool == "attn" else MaxPool(dim)

    def forward(self, vol, mask=None):
        """vol: (B, D, H, W) en [0, 1]; mask: (B, D) True en cortes reales (relleno = False)."""
        B, D = vol.shape[:2]
        if mask is None:
            mask = torch.ones(B, D, dtype=torch.bool, device=vol.device)
        x = to_25d(vol)[mask]                        # solo cortes reales: (N, 3, H, W)
        feats = self.encoder(x)
        h = feats.new_zeros(B, D, feats.shape[1])
        h[mask] = feats
        logits, att = self.pool(self.drop(h), mask)
        return logits, att


def count_params(m, trainable=False):
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable)
