# Pregunta 1 — Clasificación multietiqueta de anomalías de rodilla (RSNA Knee)

Plan completo: [PLAN.md](PLAN.md).

## Datos
Descarga `rsna-knee-abnormality-detection.zip` desde Kaggle (aceptando las reglas)
y colócalo en la raíz del repositorio. **No se descomprime**: los scripts leen
los CSV y los DICOM directamente desde el zip.

## Reproducir
```bash
pip install -r requirements.txt
cd src
python s00_extract_meta.py    # extrae solo los CSV a data/meta
python s01_catalog.py         # catálogo de series (cortes, MB) e idioma de reportes
python s02_select_series.py   # una serie sagital FS y una coronal FS por estudio
python s03_weak_labels.py     # 12 etiquetas débiles + confianza desde el reporte (9 idiomas)
python s03b_validate_weak_labels.py   # acuerdo con los 58 dorados (+IC bootstrap) -> reports/
python s04_sample_split.py    # 600 estudios estratificados + 58 dorados -> splits/
python s05_extract_images.py --view sag   # lee solo esas series del zip -> data/processed/sag
python s05_extract_images.py --view cor   # vista coronal (ablación)
python s05b_check_coherence.py --view sag # chequeos + mosaico reports/qa_montage_sag.png
python s07_pack_colab.py      # colab_bundle.zip: código + datos procesados (≈1,7 GB)
```

## Entrenar (Colab Pro)
1. Subir `colab_bundle.zip` a `MyDrive/knee/` (carpeta privada).
2. Abrir `notebooks/train_colab.ipynb` en Colab con GPU y ejecutar todas las celdas.
   Cada corrida (`A_base`, `B_base` y ablaciones) guarda en `MyDrive/knee/runs/<nombre>/`
   el historial, los pesos y las predicciones sobre `weak_val` y `gold_val`.

También corre local con GPU: `python train.py --arch resnet50 --out ../runs/B_base`.
Prueba rápida en CPU (6 estudios, 2 épocas): `python train.py --smoke --arch scratch`.

> No abras ni guardes los CSV de `data/meta` con Excel: cambia el separador a `;`.
> Si ocurre, vuelve a correr `s00_extract_meta.py`.
