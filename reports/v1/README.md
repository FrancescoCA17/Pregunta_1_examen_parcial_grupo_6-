# Primer intento (v1, Colab G4, 29/10/2026)

Configuración v1: B reentrenaba **toda** la ResNet-50 con LR 1e-4 tras 2 épocas solo-cabeza
(15 épocas, paciencia 6); A 30 épocas (paciencia 10). Resultados copiados de
`MyDrive/knee/runs/` y `MyDrive/knee/reports/`.

## Diagnóstico (solo con datos de entrenamiento y weak_val)
- Pérdida de un predictor constante óptimo con `pos_weight` en weak_train: **1,0896**.
- B: pérdida de entrenamiento 1,08–1,10 en todas las épocas, es decir, al nivel del
  predictor constante. Su mejor AUC en weak_val (0,710) fue en la época 1, aún solo con
  la cabeza; al descongelar la red (época 2) cayó a 0,540 y no se recuperó.
  Parada temprana en la época 7.
- A: la pérdida bajó apenas de 1,085 a 1,022 en 30 épocas y la AUC en weak_val seguía
  subiendo al final (0,58 → 0,70): le faltó entrenamiento.
- Se descartó la precisión bf16 como causa (en CPU, fp32 y bf16 aprenden igual).

Cambios en v2: B reentrena solo `layer3`/`layer4` con LR 2e-5 tras 3 épocas solo-cabeza
(20 épocas, paciencia 8); A 60 épocas (paciencia 15); se registra la AUC en un
subconjunto fijo de entrenamiento para distinguir subajuste de sobreajuste.

## Evaluación v1 en gold_val (n = 58)

| Corrida | AUC macro [IC 95 %] | P / R / F1 macro |
|---|---|---|
| A_base | 0,679 [0,612–0,738] | 0,467 / 0,671 / 0,506 |
| B_base | 0,649 [0,597–0,697] | 0,441 / 0,565 / 0,462 |
| texto_s03 | 0,812 [0,777–0,847] | — |
| prevalencia | 0,500 | — |

Diferencia A − B: +0,030 [−0,030; +0,087], no concluyente.

AUC por etiqueta (A / B): ACL 0,581/0,672 · MCL 0,556/0,440 · Menisco medial 0,564/0,597 ·
Menisco lateral 0,765/0,711 · OA medial 0,764/0,764 · OA lateral 0,696/0,689 ·
OA PF 0,772/0,707 · Derrame 0,862/0,826 · Sinovitis 0,728/0,595 · Baker 0,549/0,549 ·
Contusión 0,714/0,598 · Fractura 0,594/0,642.

## Historial

B_base (época, pérdida, AUC weak_val): 0 1,093 0,646 · 1 1,083 0,710 · 2 1,094 0,540 ·
3 1,102 0,620 · 4 1,091 0,558 · 5 1,091 0,568 · 6 1,090 0,640 · 7 1,091 0,654

A_base: época 0 1,085 0,635 · 5 1,097 0,581 · 10 1,081 0,618 · 15 1,067 0,650 ·
20 1,054 0,679 · 25 1,030 0,699 · 29 1,022 0,699
