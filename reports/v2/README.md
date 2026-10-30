# Segunda versión (v2, Colab G4, 29/10/2026)

Cambios frente a v1 (decididos solo con entrenamiento y weak_val): B reentrena `layer3`/`layer4`
con LR 2e-5 tras 3 épocas solo-cabeza (20 épocas, paciencia 8); A 60 épocas (paciencia 15);
se registra la AUC en 90 estudios fijos de entrenamiento (`train_auc_weak`).

## Evaluación en gold_val (n = 58), `eval_summary.txt`

| Corrida | AUC macro [IC 95 %] | P / R / F1 macro |
|---|---|---|
| **B_base** | **0,715** [0,656–0,771] | 0,483 / 0,613 / 0,505 |
| A_base | 0,654 [0,592–0,710] | 0,369 / 0,639 / 0,442 |
| texto_s03 | 0,812 [0,777–0,847] | — |
| prevalencia | 0,500 | — |

A − B = −0,062 [−0,114; −0,008]: el IC pareado no cruza 0, B supera a A.

## Curvas (`B_base_history.csv`, `A_base_history.csv`)
- **B:** la pérdida baja de 1,09 a 0,27 y la AUC de entrenamiento sube de 0,66 a 0,98: ahora
  aprende. La AUC en weak_val llega a su máximo (0,737) en la época 9 y luego baja levemente
  mientras el entrenamiento sigue subiendo (sobreajuste a las etiquetas ruidosas); se usa el
  checkpoint de la época 9. Parada temprana en la época 17.
- **A:** subajuste. Tras 60 épocas la AUC de entrenamiento es solo 0,72 (pérdida 1,08 → 1,02);
  mejor weak_val 0,696 en la época 48. Desde cero, 510 estudios con 19 % de ruido no bastan
  para aprender buenas características; la transferencia desde ImageNet sí las aporta.

## Ablaciones, desbalance y semillas (30/10/2026)

Todas cambian UN factor frente a `B_base`. Δ = diferencia de AUC macro en gold con IC 95 %
por bootstrap pareado (`eval_summary.txt`).

| Corrida | Factor | AUC macro | Δ vs B_base [IC 95 %] | Lectura |
|---|---|---|---|---|
| B_seed1 / B_seed2 | semilla | 0,722 / 0,712 | +0,006 / −0,004 | variación por azar ≈ ±0,01 |
| B_focal | pérdida focal | 0,722 | +0,007 [−0,025; 0,035] | sin diferencia |
| B_conf | ponderar por confianza | 0,713 | −0,003 [−0,026; 0,016] | sin diferencia |
| **B_hc** | solo alta confianza | **0,672** | **−0,044 [−0,087; −0,005]** | **peor**: descarta demasiada supervisión |
| B_nopw | sin `pos_weight` | 0,718 | +0,003 [−0,034; 0,037] | sin diferencia en AUC |
| B_all | reentrenar toda la ResNet (LR 2e-5) | 0,718 | +0,003 [−0,024; 0,027] | sin diferencia: el colapso de v1 fue por LR 1e-4 |
| **B_cor** | plano coronal | **0,749** | +0,034 [−0,016; 0,081] | mejor valor, no concluyente; complementario por hallazgo |

Observaciones:
- Variación entre semillas por hallazgo grande (MCL 0,54–0,71; contusión 0,64–0,76) y aún
  mayor en el recall al umbral (contusión 0,21–0,79): diferencias por hallazgo menores a eso
  no son interpretables con 58 dorados.
- Coronal frente a sagital: mejor en quiste de Baker (0,92 vs 0,65), fractura (0,82 vs 0,70),
  artrosis medial (0,94 vs 0,84) y PF; peor en LCA (0,63 vs 0,71) y LCM (0,57 vs 0,71).
  Las vistas son complementarias (fusión = trabajo futuro). Coronal entrenó con 492 estudios.
- El modelo de imagen supera a su "maestro" de texto en derrame (0,89–0,95 vs 0,68),
  artrosis medial (0,84 vs 0,80) y sinovitis (0,68 vs 0,64): hallazgos que el radiólogo
  sub-reporta en el texto pero se ven en la imagen (coherente con s03b: derrame κ 0,31).
  En macro, el texto sigue arriba: +0,096 [0,036; 0,157] frente a B_base.
- E6: `pos_weight` no cambia la AUC (es una métrica de orden, el peso actúa sobre la
  calibración). En recall de raros el efecto es mixto (fractura 0,67 → 0,83 sin peso;
  contusión 0,79 → 0,05) y del mismo tamaño que la variación entre semillas.

## Parada temprana (historiales `*_history.csv`)

El mensaje "Parada temprana: 8 épocas sin mejorar" repite el parámetro `patience=8`: por
construcción, una corrida se detiene exactamente 8 épocas después de su mejor época.
Lo informativo es CUÁNDO llega el mejor punto:

| Corrida | Mejor época | Mejor AUC weak_val | Épocas corridas | AUC train final |
|---|---|---|---|---|
| B_base | 9 | 0,737 | 18 | 0,983 |
| B_focal | 9 | 0,744 | 18 | 0,992 |
| B_cor | 10 | 0,783 | 19 | 0,978 |
| B_conf | 9 | 0,748 | 18 | 0,982 |
| B_hc | 10 | 0,699 | 19 | 0,922 |
| B_nopw | 9 | 0,735 | 18 | 0,992 |
| B_all | 12 | 0,746 | 20 (tope, sin parada) | 0,981 |
| B_seed1 | 8 | 0,726 | 17 | 0,980 |
| B_seed2 | 8 | 0,747 | 17 | 0,982 |

Todas alcanzan su mejor validación entre las épocas 8 y 12, cuando la AUC de entrenamiento
ronda 0,92–0,95; después siguen subiendo en entrenamiento (→ 0,98–0,99) y bajan en
validación: memorizan el ruido de las etiquetas débiles (las redes aprenden primero los
patrones generales y luego memorizan los casos ruidosos; Arpit et al., 2017). Es sistemático
porque todas comparten datos, cronograma (3 épocas de cabeza + one-cycle) y nivel de ruido.

## Grad-CAM: interpretabilidad (entregable 5, 30/10/2026)

Figuras en `MyDrive/knee/reports_v2/gradcam/{B_base,A_base}/` (copia local en
`reports/gradcam/`, fuera de git). Por hallazgo: los 3 dorados positivos con mayor
probabilidad (TP) y el falso positivo más seguro (FP), en el corte que eligió la atención.

### Modelo B (ResNet-50), revisión visual caso por caso

| Hallazgo | AUC | Dónde cae el mapa | ¿Plausible? |
|---|---|---|---|
| **Derrame** | 0,89 | 3/3 TP sobre el líquido del receso suprapatelar (la colección brillante anterior al fémur). El FP mira la región poplítea. | ✅ Sí |
| **Contusión** | 0,76 | 3/3 TP dentro del cóndilo femoral, sobre el edema óseo moteado. El FP mira hueso subcondral sin lesión visible. | ✅ Sí |
| **LCA** | 0,71 | Ningún TP en la escotadura intercondílea. Los cortes son laterales (18/21, 15/40, 7/28) y el calor cae en el derrame suprapatelar o en la grasa y el hueso anteriores. | ⚠️ Signos secundarios |
| **Menisco lateral** | 0,64 | Tejido blando posterior, región suprapatelar o periferia. Nunca el menisco (el triángulo oscuro entre fémur y tibia). | ❌ No |
| **Menisco medial** | 0,60 | Los 4 casos usan cortes del extremo de la pila (3/24, 3/30, 30/33, 31/35), fuera de la articulación: grasa subcutánea y músculo. | ❌ No |

- **Lectura:** B es anatómicamente coherente justo en los hallazgos que más
  "brillan" en una secuencia sensible a líquido con supresión de grasa (líquido libre,
  edema óseo), y donde mejor rinde.
- Para estructuras pequeñas y finas (meniscos, ligamentos) no localiza la lesión. Usa:
  - **correlaciones:** P(derrame | LCA) = 0,75 frente a 0,60 global en los dorados;
  - **contexto periférico:** edema de partes blandas.
- Esto concuerda con su menor AUC en meniscos y con la atención por corte: LCA en el
  tercio central solo el 2 %; meniscos en extremos opuestos el 43 %, cerca del azar.
- Con mapas de 7×7 y cortes de 256 px reducidos a 224, un menisco ocupa menos de una celda
  del mapa.

### Modelo A (desde cero)

- Probabilidades bajas (0,65–0,81) y mapas poco específicos: en el mismo estudio, el mapa
  de "menisco medial" y el de "derrame" son casi idénticos.
- La atención se concentra en el centro de la pila para casi todos los hallazgos.
- A aprendió un detector genérico de "rodilla anormal", coherente con su subajuste
  (AUC de entrenamiento 0,72).

### Implicancia clínica (para la discusión)
Un modelo que detecta la rotura de LCA por el derrame fallaría en una rotura sin derrame y
daría falsos positivos en derrames sin rotura. Las métricas globales (AUC 0,71) no muestran
ese riesgo; Grad-CAM sí. Mejoras posibles:
- más resolución en la región articular (recorte centrado en la articulación);
- supervisión por corte;
- fusión con la vista coronal, donde B rinde mejor en estructuras medial/lateral.
