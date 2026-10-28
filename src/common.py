"""Rutas y constantes compartidas por todos los scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "rsna-knee-abnormality-detection.zip"
META = ROOT / "data" / "meta"
INTERIM = ROOT / "data" / "interim"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
SEED = 42
