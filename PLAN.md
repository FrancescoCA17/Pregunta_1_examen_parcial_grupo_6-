# Plan de trabajo — Pregunta 1: Clasificación multietiqueta de anomalías de rodilla (RSNA)

**Entrega:** 31 de octubre de 2026, en clase. Informe en PDF (formato NeurIPS, 6–10 páginas) y repositorio de GitHub que corra de extremo a extremo con **un solo comando** documentado en el README, más `requirements.txt`. Las entregas tardías pierden un 10 % por día.
**Fecha de este plan:** 27/10/2026. Quedan **4 días**, así que el plan está ajustado a ese plazo.

---

## 1. Lo que ya sabemos de los datos (paso 1, terminado)

Todo se obtuvo leyendo el índice del zip, sin descomprimirlo (`src/s00`–`s02`).

| Hecho | Valor | Consecuencia |
|---|---|---|
| Tamaño | 264,6 GB comprimido / 569,8 GB descomprimido, 819 635 DICOM | No se descomprime. Los DICOM se leen directamente desde el zip. |
| Estudios de train | 4407, cada uno con su reporte | Población de la que sale la muestra. |
| Estudios dorados | **58** (etiquetas 0/1) | Conjunto de **validación limpio**, como pide el enunciado. Nunca se entrena con ellos. |
| Etiquetas | 12 columnas de `train.csv`, en el mismo orden que `sample_submission.csv` | Coinciden con la pestaña "Data". |
| Plano/secuencia | Vienen en `train_series.csv` (`Anatomical_Plane`, `Fat_Suppression`; `Fluid_Sensitive` es idéntica a `Fat_Suppression`) | No hace falta leer 800 mil cabeceras. |
| Idiomas (langdetect) | en 1736, es 682, tr 546, hr 406, el 321, de 262, bg 220, nl 153, fr 81 | Hay 9 idiomas. Francés no tiene ningún dorado. Inglés concentra 28 de los 58 dorados. |
| Cortes por serie | Mediana ≈30. Hay series 3D isotrópicas de hasta 320 cortes. | Las 2D se prefieren. Las 3D se submuestrean a unos 30 cortes. |
| Cobertura sagital con supresión de grasa | 94,2 % de los estudios, **56/58 dorados** | Es la vista principal. |
| Cobertura coronal con supresión de grasa | 96,4 % de los estudios, 56/58 dorados | Es la vista para la ablación. |
| Fabricantes | GE, Siemens, Philips, Toshiba, con matrices de 384 a 1024 px | Se normaliza la intensidad por percentiles y se redimensiona a un tamaño fijo. |

Positivos entre los 58 dorados: ACL 24 · **MCL 9** · Men. med. 26 · Men. lat. 23 · OA med. 15 · OA lat. 11 · OA PF 21 · Derrame 35 · Sinovitis 27 · **Baker 12** · Contusión 19 · Fractura 18.
Con 9 positivos de MCL, su AUC tendrá un intervalo de confianza muy ancho. Por eso se reporta con *bootstrap*.

Los demás detalles están en `data/meta/series_catalog.csv`, `study_catalog.csv` y `study_series.csv`.

---

## 2. Decisiones de diseño (justificadas para el informe)

1. **Reducción de escala:** se usa una serie sagital con supresión de grasa por estudio (DP/T2 FS, sensible a líquido), que es la recomendación del enunciado. La coronal con supresión de grasa se usa solo en la ablación. La muestra de entrenamiento tiene **≈500 estudios** con etiqueta débil, dentro del rango 300–600 que pide el enunciado.
2. **Extracción selectiva:** solo se leen del zip las series elegidas, con `zipfile` + `pydicom`. Son unos 4–5 GB comprimidos en lugar de 570 GB.
3. **Representación de la imagen:** paquetes 2.5D de 3 cortes contiguos, de 224×224, normalizados por percentiles 1–99. Se guardan como `uint8` en *shards* `.npz` comprimidos.
4. **Del corte al estudio:** se calcula un vector de 12 *logits* por corte y se agregan por estudio con **max-pooling** o **atención**. La agregación con atención también es una opción de ablación.
5. **Particiones (sin fuga):**
   - `gold_val`: los 58 dorados (56 con vista sagital). Se usan **solo para la evaluación final**.
   - `weak_train` (≈85 %) y `weak_val` (≈15 %): estudios con etiqueta débil. `weak_val` sirve para *early stopping*, elegir el umbral y comparar ablaciones.
   - El split es por estudio. El dataset no trae un ID de paciente; esto se declara como limitación.
6. **Umbral de decisión:** se elige por etiqueta maximizando F1 en `weak_val`, y luego se reporta su precisión, recall y F1 en `gold_val`. Así los dorados no se usan para ajustar nada.

7. **Etiquetas débiles para entrenar:** se usa **U-zeros** (lo incierto cuenta como 0) para la etiqueta dura. La exactitud real de las menciones inciertas fue de solo 0,36. Además se guarda el objetivo blando `<L>_p` para la ablación "ponderar por confianza". La confianza asignada está sobreestimada en los positivos (asignada 0,95, real 0,76), y eso se discute en el informe.
8. **Hallazgo clave del análisis de errores:** en sinovitis y derrame, buena parte de los errores son menciones que el reporte no hace o que contradice. Los dorados parecen anotados sobre la imagen, así que el ruido de esas etiquetas no se puede reducir solo con el texto. Esto justifica el manejo de etiquetas ruidosas (E4).
9. **DICOM comprimidos:** la pestaña "Data" indica JPEG Lossless o JPEG 2000. Para leerlos, `s05` necesita `pylibjpeg`, `pylibjpeg-libjpeg` y `pylibjpeg-openjpeg`.

---

## 3. Pasos, entregables y responsables

Roles sugeridos para los 5 integrantes:
- **A**: datos e imágenes
- **B**: NLP y etiquetas débiles
- **C**: modelo A (CNN desde cero)
- **D**: modelo B (transfer learning) y ablaciones
- **E**: evaluación, Grad-CAM e informe

| # | Paso | Detalle | Entregable | Resp. | Día |
|---|---|---|---|---|---|
| 1 | Exploración | ✅ Hecho (`s00`, `s01`): catálogo de series e idiomas | Base | A | 27 |
| 2 | Selección de serie | ✅ Hecho (`s02`): una sagital FS y una coronal FS por estudio, con la 3D como respaldo | Ablación (4) | A | 27 |
| 3 | Etiquetas débiles (`s03`) | ✅ Hecho. Reglas multilingües que siguen las definiciones de la pestaña "Data". Normalización Unicode, eliminación de la sección clínica, separación de la conclusión, negación al estilo NegEx (antes y después del término), incertidumbre y confianza por tipo de evidencia. Las reglas se construyeron y auditaron **solo con reportes no dorados** (3 rondas de auditoría manual). | E1 | B | 28 |
| 4 | Validación de etiquetas (`s03b`) | ✅ Hecho, con la v1 congelada. Exactitud 0,81 (**ruido 19 %**), F1 macro 0,72, κ macro 0,56, AUC macro del objetivo blando 0,81. Mejores etiquetas: ACL (F1 0,89), meniscos (0,80–0,84) y Baker (0,76). Peores: sinovitis (recall 0,44), PF OA (recall 0,48) y derrame (κ 0,31). Resultados en `reports/`. | E1, E7 | B | 28 |
| 5 | Muestreo estratificado (`s04`) | ≈500 estudios no dorados que tengan sagital FS. Muestreo iterativo multietiqueta para garantizar al menos 30 positivos débiles por hallazgo, más una cuota por idioma. Dividir en `weak_train`/`weak_val`. | E3, E6 | A | 28 |
| 6 | Extracción de imágenes (`s05`) | Leer las series desde el zip, ordenar los cortes por `ImagePositionPatient`, normalizar por percentiles, redimensionar a 224 y quedarse con 16 cortes centrales (o todos si hay menos). Guardar en *shards* `.npz`. Incluye los dorados y la vista coronal para la ablación. Se ejecuta **en local**, en segundo plano. | Datos | A | 28 |
| 7 | Desbalance | Prevalencia débil de cada etiqueta en la muestra. Mitigación: `pos_weight` en BCE **y** pérdida focal, evaluadas de forma comparada. | E6 | D | 28 |
| 8 | Modelo A: CNN desde cero | 4–5 bloques conv-BN-ReLU con conexiones residuales, entrada 2.5D (3 canales), 12 *logits* y agregación por estudio. | E2 | C | 28–29 |
| 9 | Modelo B: transfer learning | ResNet-50 de ImageNet (`torchvision`) con una cabeza propia de 12 sigmoides y la misma agregación. Fine-tuning en dos etapas: 1) solo la cabeza, 2) descongelar `layer3`/`layer4` con un LR más bajo. | E2 | D | 29 |
| 10 | Evaluación | AUC macro sobre 12 etiquetas (métrica oficial) y AUC por etiqueta en `gold_val`, con IC 95 % por *bootstrap* (1000 remuestreos). Precisión/recall/F1 al umbral elegido en `weak_val`. Incluir una línea base trivial que predice la prevalencia. | E3 | E | 29–30 |
| 11 | Ablación (≥2 factores) | (a) BCE vs. pérdida focal; (b) ruido: ponderar por confianza vs. usar solo alta confianza; (c, si hay tiempo) sagital vs. coronal. Se hace con el modelo B y pocas épocas. | E4 | D | 29–30 |
| 12 | Grad-CAM | Implementarlo con *hooks* de PyTorch sobre `layer4` y el último bloque de la CNN. Cubrir al menos 3 hallazgos (menisco medial, ACL, derrame) usando casos dorados positivos. Comentar si la activación cae en la región anatómica correcta. | E5 | E | 30 |
| 13 | Sesgo lingüístico | Media página basada en el desglose por idioma del paso 4: francés sin dorados, idiomas con pocos ejemplos y el efecto en un despliegue real. | E7 | B | 30 |
| 14 | Repositorio | `run_all` con un solo comando, README, `requirements.txt`, semillas fijas y CSV de particiones versionados. | Entrega | A | 30 |
| 15 | Informe NeurIPS | Motivación, ≥5 referencias, metodología, experimentos, resultados y limitaciones. | Entrega | todos | 30–31 |

---

## 4. Dónde corre cada cosa

| Etapa | Dónde | Por qué |
|---|---|---|
| `s00`–`s05` (catálogo, etiquetas, muestreo, extracción) | Local, junto al zip | El zip de 265 GB no se puede subir. |
| Subida | `data/processed/*.npz` (unos cientos de MB) → Google Drive | |
| Entrenamiento, ablación, Grad-CAM | Colab Pro (GPU) | Copiar los *shards* a `/content` antes de entrenar, guardar *checkpoints* en Drive en cada época y usar `torch.autocast`. |

---

## 5. Referencias candidatas (≥5)

1. Bien et al., 2018. *Deep-learning-assisted diagnosis for knee MRI: MRNet*. PLOS Medicine.
2. He et al., 2016. *Deep Residual Learning for Image Recognition*. CVPR.
3. Lin et al., 2017. *Focal Loss for Dense Object Detection*. ICCV.
4. Selvaraju et al., 2017. *Grad-CAM*. ICCV.
5. Chapman et al., 2001. *NegEx: a simple algorithm for identifying negated findings*. J. Biomed. Inform.
6. Irvin et al., 2019. *CheXpert* (etiquetador de reportes con incertidumbre). AAAI.
7. Ratner et al., 2017. *Snorkel: weak supervision*. VLDB.
8. Ilse et al., 2018. *Attention-based Deep Multiple Instance Learning*. ICML.

---

## 6. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| El tiempo (4 días) | Priorizar E1–E3 y E6 antes que la ablación con coronal. Entrenar con pocas épocas. |
| Reglas débiles en idiomas con pocos ejemplos | Reportarlo en E7. Ponderar por confianza. |
| Sobreajuste a los dorados | Los dorados no se usan para ajustar reglas, umbrales ni hiperparámetros. |
| Sesiones de Colab que se caen | *Checkpoints* en Drive y datos en *shards*. |
| Licencia de Kaggle | Revisar "Rules". No subir datos ni CSV al repositorio (`.gitignore`). |
