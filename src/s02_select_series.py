"""Paso 2: elige una serie por estudio para cada vista usada.

- Vista principal: sagital con supresión de grasa (DP/T2 FS, sensible a líquido).
- Vista de ablación: coronal con supresión de grasa.
- Respaldo sin supresión de grasa (sag_nofs / cor_nofs): solo se usa en la
  extracción para dorados sin serie FS, así la evaluación cubre los 58.
Se prefieren series 2D (MIN_SLICES..MAX_SLICES cortes); una adquisición 3D
isotrópica (> MAX_SLICES) solo se usa si el estudio no tiene otra, y en la
extracción se submuestrea a ~TARGET cortes. Entre varias candidatas se toma la
de número de cortes más cercano a la mediana (≈30).
"""
import pandas as pd

from common import META

MIN_SLICES, MAX_SLICES, TARGET = 15, 60, 30
VIEWS = {"sag_fs": ("Sagittal", 1), "cor_fs": ("Coronal", 1),
         "sag_nofs": ("Sagittal", 0), "cor_nofs": ("Coronal", 0)}


def pick(series, plane, fat_sup):
    cand = series[(series.Anatomical_Plane == plane)
                  & (series.Fat_Suppression == fat_sup)
                  & (series.n_slices >= MIN_SLICES)].copy()
    cand["is_3d"] = cand.n_slices > MAX_SLICES
    cand["dist"] = (cand.n_slices - TARGET).abs()
    cand = cand.sort_values(["StudyInstanceUID", "is_3d", "dist", "SeriesInstanceUID"])
    return cand.groupby("StudyInstanceUID").first()


def main():
    series = pd.read_csv(META / "series_catalog.csv")
    studies = pd.read_csv(META / "study_catalog.csv").set_index("StudyInstanceUID")
    for view, (plane, fs) in VIEWS.items():
        chosen = pick(series, plane, fs)
        studies[f"{view}_series"] = chosen.SeriesInstanceUID
        studies[f"{view}_slices"] = chosen.n_slices
        studies[f"{view}_mb"] = chosen.zip_mb
        studies[f"{view}_is3d"] = chosen.is_3d
    studies.reset_index().to_csv(META / "study_series.csv", index=False)

    for view in VIEWS:
        has = studies[f"{view}_series"].notna()
        print(f"{view}: {has.mean():.1%} de estudios, "
              f"oro con serie: {(has & studies.gold).sum()}/58, "
              f"mediana cortes {studies[f'{view}_slices'].median():.0f}, "
              f"3D usadas: {int(studies[f'{view}_is3d'].fillna(False).sum())}")
    both = studies.sag_fs_series.notna() & studies.cor_fs_series.notna()
    print(f"ambas vistas: {both.mean():.1%}")


if __name__ == "__main__":
    main()
