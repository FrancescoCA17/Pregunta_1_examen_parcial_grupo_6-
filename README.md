# Pregunta 1 — Clasificación multietiqueta de anomalías de rodilla (RSNA Knee)

Clasificación de 12 hallazgos en resonancia magnética de rodilla con CNN y supervisión
débil: las etiquetas de entrenamiento se extraen de reportes en 9 idiomas y se validan
contra los 58 estudios con etiqueta dorada. Plan y decisiones: [PLAN.md](PLAN.md).
Resultados: [reports/v2/README.md](reports/v2/README.md).

## Datos
Descarga `rsna-knee-abnormality-detection.zip` desde Kaggle (aceptando las reglas de la
competencia) y colócalo en la raíz del repositorio. **No se descomprime**: los scripts leen
los CSV y los DICOM directamente desde el zip y extraen solo las series necesarias
(≈11 GB de 265 GB). Los datos y las imágenes derivadas no se incluyen en el repositorio.

## Reproducir con un solo comando
```bash
pip install -r requirements.txt
python run_all.py
```
`run_all.py` ejecuta las etapas en orden y se puede reanudar: si se interrumpe, al
volver a lanzarlo omite los shards ya extraídos y retoma cada corrida desde su último
*checkpoint*.

| Etapa | Qué hace | Dónde |
|---|---|---|
| `data` | `s00`–`s05b`: extrae los CSV, arma el catálogo, elige las series, genera las etiquetas débiles y las valida contra los dorados, hace el muestreo estratificado, extrae las imágenes y corre los chequeos | Local, junto al zip (~25 min) |
| `train` | 10 corridas: modelos A y B, ablaciones y semillas → `runs/<nombre>/` | GPU (~3 min por corrida en Colab G4) |
| `eval` | `evaluate.py`: AUC con IC, P/R/F1 y ablaciones → `reports/eval_*` | CPU, segundos |
| `gradcam` | `gradcam.py` para B_base y A_base → `reports/gradcam/` | GPU o CPU |
| `pack` | `colab_bundle.zip` / `colab_code.zip` para Colab (opcional) | Local |
| `report` | Figuras del informe y compilación de `report/main.tex` (NeurIPS 2025) con `latexmk` | Local, ~1 min |

Ejemplos:
- `python run_all.py --stages data` corre solo la etapa de datos.
- `python run_all.py --stages train,eval --runs B_base A_base` entrena y evalúa solo esas dos corridas.
- `python run_all.py --dry-run` muestra los comandos sin ejecutarlos.

Prueba rápida del entrenamiento en CPU (6 estudios, 2 épocas):
`cd src && python train.py --smoke --arch scratch`.

## Entrenar en Colab (sin GPU local)
1. Corre `python run_all.py --stages data,pack` y sube `colab_bundle.zip` a `MyDrive/knee/`
   (carpeta privada). Si después solo cambia el código, basta subir `colab_code.zip`.
2. Abre `notebooks/train_colab.ipynb` en Colab con GPU y ejecuta las celdas de montaje,
   entrenamiento, evaluación y Grad-CAM. Los resultados quedan en `MyDrive/knee/runs_v2/`
   y `MyDrive/knee/reports_v2/`.

## Estructura
```
run_all.py            comando único
src/                  s00–s07 (datos), models.py, train.py, evaluate.py, gradcam.py
notebooks/            train_colab.ipynb
splits/               IDs de estudio por partición (weak_train / weak_val / gold_val)
report/               informe NeurIPS (main.tex, sections/, refs.bib)
reports/              validación de etiquetas, prevalencias, coherencia, sesgo por idioma (e7_*),
                      v1/ (primer intento), v2/ (resultados finales)
```

> No abras ni guardes los CSV de `data/meta` con Excel: cambia el separador a `;`.
> Si ocurre, vuelve a correr `s00_extract_meta.py`.
