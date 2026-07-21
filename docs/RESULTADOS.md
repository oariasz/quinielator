# Resultados reproducidos

Ejecución local con Python 3.11.10, semilla 42, TensorFlow 2.21, Keras 3.15 y
scikit-learn 1.9. Se prepararon 1.068 partidos de 23 Mundiales (1930–2026) y se evaluaron
967 partidos de 1958–2026 sin entrenar con el Mundial objetivo.

## Comparación global

| Modelo | Accuracy 1/X/2 | Marcador exacto | MAE goles | Log loss | Brier |
|---|---:|---:|---:|---:|---:|
| ensemble | **51,6%** | **12,3%** | 0,985 | **1,013** | **0,606** |
| poisson | 51,2% | **12,3%** | **0,985** | 1,028 | 0,614 |
| baseline | 50,3% | 12,2% | 1,028 | 1,031 | 0,617 |
| keras | 50,8% | 10,3% | 1,162 | 1,267 | 0,657 |
| outcome | 43,0% | 10,4% | 1,096 | 1,280 | 0,722 |

El ensamble gana globalmente en accuracy, log loss y Brier, y empata con Poisson en
marcador exacto. Poisson obtiene un MAE apenas menor. La red Keras individual no es
superior: el proyecto la compara en vez de asumir que mayor complejidad implica mejor
predicción.

## Evolución reciente del ensamble

| Mundial | Historia para entrenar | Accuracy edición | Accuracy acumulada | Marcador exacto | Cobertura FIFA | Peso Poisson |
|---:|---:|---:|---:|---:|---:|---:|
| 2018 | 836 partidos | 56,2% | 52,1% | 23,4% | 100,0% | 100% |
| 2022 | 900 partidos | 56,2% | 52,4% | 9,4% | 100,0% | 75% |
| 2026 | 964 partidos | 45,2% | 51,6% | 10,6% | 0,0% | 100% |

La edición 2026 contradice una mejora monótona: entraron más datos, pero la precisión de
signos bajó. El intervalo Wilson de 95% para 47/104 es 36,0%–54,8%. Además, el baseline
Elo obtuvo 57,7% en esa edición, frente a 45,2% del ensamble. No se cambia de modelo a
posteriori, porque hacerlo después de ver 2026 sería selección con el conjunto de prueba.

Globalmente, el error de calibración del ensamble fue 0,112, frente a 0,091 del baseline.
La mejora de accuracy y log loss no elimina la necesidad de calibrar mejor las
probabilidades.

## Tabla completa de 2026

El reporte contiene los 104 juegos, predicción y resultado real a 90 minutos. Verde
significa signo acertado y rojo, signo fallado. La evaluación separa marcador exacto del
signo; cuatro tandas posteriores están señaladas, pero no cuentan para 1/X/2.

- [Tabla HTML a color](resultados-2026/mundial_2026_ensemble.html)
- [Tabla Markdown](resultados-2026/mundial_2026_ensemble.md)
- [CSV auditable](resultados-2026/mundial_2026_ensemble.csv)
- [Resumen visual SVG](resultados-2026/mundial_2026_ensemble_resumen.svg)

Distribución predicha: 30 signos `1`, 31 signos `X` y 43 signos `2`. Distribución real:
52 signos `1`, 29 signos `X` y 23 signos `2`. El modelo subestimó las victorias del
primer equipo bajo la orientación prepartido del proyecto.

## Ejemplo: final de 2026

Con 964 partidos anteriores, el procedimiento eligió 100% Poisson. Para Argentina–España
predijo marcador modal 0–0, pero signo `2`: 27,1% Argentina, 34,8% empate y 38,1% España.
El resultado real fue 0–0 a 90 minutos y España ganó 1–0 después de prórroga. Acertó el
marcador exacto, falló el signo y acertó quién avanzó. Esto ilustra por qué el signo
agregado puede diferir de la clase del marcador modal.

## Qué influyó en 2026

El análisis de permutación entrenó con los 964 partidos anteriores y midió los 104 de
2026, con cinco repeticiones y log loss base 1,0031.

| Familia | Δ log loss | Parte del aporte positivo |
|---|---:|---:|
| Experiencia | 0,0403 | 37,7% |
| Contexto del partido | 0,0226 | 21,1% |
| Forma y resultados | 0,0219 | 20,5% |
| Goles recientes | 0,0116 | 10,8% |
| Descanso | 0,0076 | 7,1% |
| Elo | 0,0028 | 2,6% |
| Ranking FIFA | 0,0002 | 0,2% |
| Enfrentamientos directos | -0,0022 | 0,0% |

Las variables individuales líderes fueron confederación UEFA del segundo equipo,
partidos históricos del segundo equipo, condición de anfitrión, indicadores CONCACAF y
AFC del segundo equipo y experiencia en eliminatorias. No implican causalidad. La
codificación one-hot evita interpretar las confederaciones como números ordinales. La
escasa importancia FIFA es coherente con 0% de cobertura válida en 2026.

- [Informe completo de variables](resultados-2026/features_ensemble_2026.md)
- [Gráfico SVG](resultados-2026/features_ensemble_2026.svg)
- [CSV por variable](resultados-2026/features_ensemble_2026.csv)
- [CSV por familia](resultados-2026/features_ensemble_2026_groups.csv)

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
python -m quinielator report --model ensemble --world-cup 2026 --publish-docs
python -m quinielator analyze-features --model ensemble --world-cup 2026 --repeats 5 --publish-docs
```

Los datos, modelos y reportes generales no se versionan. Las piezas didácticas de 2026
sí se copian a `docs/resultados-2026/`. Actualizar fuentes o dependencias puede cambiar
las cifras; [FUENTES_DE_DATOS.md](FUENTES_DE_DATOS.md) conserva las huellas exactas de
esta ejecución.
