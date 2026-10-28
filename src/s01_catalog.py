"""Paso 1: catálogo de series (cortes y tamaño leídos del índice del zip)
e idioma de cada reporte. No lee píxeles ni descomprime nada."""
import collections
import zipfile

import pandas as pd
from langdetect import DetectorFactory, detect

from common import LABELS, META, ZIP_PATH

DetectorFactory.seed = 0


def detect_lang(text):
    try:
        return detect(text.replace("[DATE]", ""))
    except Exception:
        return "unk"


def main():
    series = pd.read_csv(META / "train_series.csv")
    train = pd.read_csv(META / "train.csv")

    n_slices, size = collections.Counter(), collections.Counter()
    first_file = {}
    with zipfile.ZipFile(ZIP_PATH) as z:
        for info in z.infolist():
            parts = info.filename.split("/")
            if parts[0] == "train_series" and len(parts) == 4:
                n_slices[parts[2]] += 1
                size[parts[2]] += info.compress_size
                first_file.setdefault(parts[2], info.filename)
    series["n_slices"] = series.SeriesInstanceUID.map(n_slices)
    series["zip_mb"] = (series.SeriesInstanceUID.map(size) / 1e6).round(2)
    series["first_file"] = series.SeriesInstanceUID.map(first_file)
    series.to_csv(META / "series_catalog.csv", index=False)

    studies = train[["StudyInstanceUID"]].copy()
    studies["lang"] = train.Report.map(detect_lang)
    studies["gold"] = train[LABELS].notna().any(axis=1)
    studies.to_csv(META / "study_catalog.csv", index=False)

    print(series.groupby(["Anatomical_Plane", "Fat_Suppression"]).size())
    print(pd.crosstab(studies.lang, studies.gold, margins=True))


if __name__ == "__main__":
    main()
