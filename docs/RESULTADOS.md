# Resultados reproducidos

Ejecución local con Python 3.11.10, semilla 42, TensorFlow 2.21, Keras 3.15 y
scikit-learn 1.9. Se prepararon 1.068 partidos de 23 Mundiales (1930–2026) y se evaluaron
967 partidos de 1958–2026.

## Comparación global

| Modelo | Accuracy 1X2 | Marcador exacto | MAE goles | Log loss | Brier |
|---|---:|---:|---:|---:|---:|
| ensemble | **51,9%** | **13,2%** | **0,974** | **1,020** | **0,609** |
| baseline | 50,3% | 12,2% | 1,028 | 1,031 | 0,617 |
| poisson | 50,1% | 12,1% | 0,990 | 1,035 | 0,619 |
| keras | 49,8% | 10,1% | 1,131 | 1,126 | 0,652 |
| outcome | 42,8% | 10,7% | 1,088 | 1,243 | 0,721 |

El ensamble supera al baseline en accuracy, marcador exacto, MAE, log loss y Brier. La
red Keras individual no es superior: el proyecto la compara en lugar de asumirlo.

## Evolución reciente del ensamble

| Mundial | Historia para entrenar | Accuracy edición | Accuracy acumulada | Marcador exacto | Cobertura FIFA |
|---:|---:|---:|---:|---:|---:|
| 2018 | 836 partidos | 57,8% | 51,2% | 17,2% | 100,0% |
| 2022 | 900 partidos | 53,1% | 51,3% | 4,7% | 100,0% |
| 2026 | 964 partidos | 56,7% | 51,9% | 16,3% | 0,0% |

La mejora acumulada de 2022 a 2026 ilustra la hipótesis, pero ediciones anteriores tienen
subidas y bajadas. La gráfica `reports/accuracy_ensemble.svg` muestra ambas curvas.
Para 2026, el intervalo Wilson de 95% del accuracy es 47,1%–65,8%; comunica mejor la
incertidumbre que un único porcentaje.

La matriz de confusión muestra que el ensamble predijo muy pocos empates. Su error de
calibración global fue 0,135, frente a 0,091 del baseline: aunque mejora accuracy y log
loss, todavía existe espacio para calibrar y representar mejor los empates.

## Ejemplo: final de 2026

Con 964 partidos anteriores, el ensamble predijo Argentina 1–0 España a 90 minutos, con
37,7% Argentina, 25,4% empate y 36,9% España. El resultado real fue 0–0 a 90 minutos y
España ganó 1–0 después de prórroga. Por tanto, la predicción no acertó el 1X2 ni el
marcador exacto; conservar también los fallos evita una demostración engañosa.

## Reproducir

```bash
python -m quinielator data download
python -m quinielator data validate
python -m quinielator data prepare
python -m quinielator evaluate --model baseline
python -m quinielator evaluate --model poisson
python -m quinielator evaluate --model outcome
python -m quinielator evaluate --model keras
python -m quinielator evaluate --model ensemble
python -m quinielator report --model ensemble
```

Los CSV y modelos no se versionan. Las cifras de este documento son la referencia de la
ejecución verificada; una actualización de las fuentes o dependencias puede cambiarlas.
