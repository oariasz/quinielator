# Modelos usados y por qué

Quinielator compara cinco enfoques bajo el mismo corte temporal. No se declara a una red
neuronal ganadora por ser más compleja: cada modelo debe demostrarlo fuera de muestra.

## Qué predice el software

Cada partido genera dos salidas relacionadas pero diferentes:

1. un marcador modal a 90 minutos, por ejemplo `1–0`;
2. probabilidades agregadas y un signo: `1` gana el primer equipo, `X` empata y `2`
   gana el segundo.

El signo es la clase con mayor probabilidad entre 1/X/2. Puede ocurrir que el marcador
individual más probable sea `0–0`, pero que la suma de todos los marcadores visitantes
sea mayor que la suma de todos los empates. Por eso marcador modal y signo no se fuerzan
a coincidir. La prórroga y la tanda no cambian ninguna de esas dos salidas.

## Comparación conceptual

| Modelo | Objetivo directo | Motivo de inclusión | Ventaja | Limitación principal |
|---|---|---|---|---|
| `baseline` | goles medios ajustados por Elo | referencia simple que los demás deben superar | transparente y estable con poca historia | solo explota Elo y promedios globales |
| `poisson` | goles de cada equipo | los goles son conteos no negativos; produce una distribución completa de marcadores | interpretable, eficiente y adecuado para muestras pequeñas | supone una forma Poisson y dos intensidades separadas |
| `outcome` | clase H/D/A | contrasta modelar el signo directamente con modelar goles | probabilidades 1/X/2 directas y regularizadas | el marcador posterior es una aproximación |
| `keras` | dos intensidades de gol | comprueba si interacciones no lineales aportan algo | puede aprender relaciones no lineales | más varianza, coste y riesgo de sobreajuste |
| `ensemble` | mezcla de intensidades/probabilidades | permite combinar los sesgos de Poisson y Keras | el peso se decide con validación temporal anterior | no mejora si ambos componentes fallan igual |

## Implementación exacta

### Baseline Elo

`BaselineModel` aprende el promedio histórico de goles del primer y segundo equipo. En
predicción los multiplica por un ajuste exponencial acotado de la diferencia Elo. Es la
vara mínima y también el fallback declarado cuando hay menos de 200 partidos previos.

### Poisson con scikit-learn

`SklearnPoissonModel` entrena dos pipelines independientes de `PoissonRegressor`: uno
para goles del primer equipo y otro para los del segundo. Cada pipeline imputa la
mediana, añade indicadores de ausencia, estandariza y usa regularización `alpha=0.7`.
La documentación oficial de scikit-learn describe
[`PoissonRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)
como un GLM con distribución Poisson y enlace logarítmico, apropiado para objetivos de
conteo no negativos.

Las intensidades se limitan a 0,05–5,00 y se convierten en una matriz de marcadores 0–8.
La cola superior se acumula en la última celda para conservar masa probabilística.

### Regresión logística 1/X/2

`SklearnOutcomeModel` usa `LogisticRegression` regularizada (`C=0.35`, hasta 2.000
iteraciones) sobre H/D/A. La
[documentación oficial](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
la define como clasificador regularizado y admite el caso multinomial. Se incluye para
medir si optimizar el signo directamente gana a estimar goles. Para presentar marcador,
transforma la diferencia de probabilidades en dos intensidades aproximadas.

### Red TensorFlow/Keras

`KerasMatchModel` comparte el mismo preprocesamiento y usa una MLP `32 → dropout 0,20 →
16 → 2`. La salida `softplus` garantiza intensidades positivas; Adam usa tasa 0,003 y la
pérdida es [`Poisson`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/Poisson).
El último 15% temporal del entrenamiento sirve como validación, `shuffle=False` conserva
el orden y [`EarlyStopping`](https://keras.io/api/callbacks/early_stopping/) restaura los
mejores pesos cuando deja de mejorar `val_loss`.

### Ensamble temporal

`EnsembleModel` entrena Poisson y Keras en el 80% inicial de la historia disponible y
elige entre pesos Poisson 0%, 25%, 50%, 75% o 100% usando log loss en el 20% final. Solo
después reajusta ambos componentes con toda la historia. El Mundial objetivo nunca
participa en esa selección.

Para predecir 2026, la validación anterior eligió **100% Poisson y 0% Keras**. Se conserva
el nombre `ensemble` porque ese fue el procedimiento predefinido; no se cambió el modelo
después de mirar el resultado de 2026.

## Resultados comparables

Backtest `strict_editions`: 967 partidos de los Mundiales 1958–2026, siempre entrenando
con ediciones anteriores.

| Modelo | Accuracy 1/X/2 | Marcador exacto | MAE goles | Log loss | Brier |
|---|---:|---:|---:|---:|---:|
| ensemble | **51,6%** | **12,3%** | 0,985 | **1,013** | **0,606** |
| poisson | 51,2% | **12,3%** | **0,985** | 1,028 | 0,614 |
| baseline | 50,3% | 12,2% | 1,028 | 1,031 | 0,617 |
| keras | 50,8% | 10,3% | 1,162 | 1,267 | 0,657 |
| outcome | 43,0% | 10,4% | 1,096 | 1,280 | 0,722 |

Accuracy y marcador exacto se maximizan; MAE, log loss y Brier se minimizan. El ensamble
es el mejor global por accuracy y calidad probabilística, pero no gana cada Mundial. En
2026 obtuvo 45,2%, por debajo del baseline de esa edición: un resultado negativo útil que
evita vender una mejora inexistente.

## Selección y reproducibilidad

- `strict_editions` fija un modelo para toda la edición; los resultados internos solo
  actualizan variables prepartido de los juegos siguientes.
- `walk_forward` permite reajustar después de cada resultado ya observado.
- semillas 42 cubren Python, NumPy, scikit-learn y TensorFlow.
- imputadores y escaladores se ajustan exclusivamente con el bloque de entrenamiento.
- las primeras ediciones usan `baseline_fallback` hasta disponer de 200 partidos.

El diseño completo está en [METODOLOGIA.md](METODOLOGIA.md) y los números reproducidos
en [RESULTADOS.md](RESULTADOS.md).
