# Metodología

## Pregunta del experimento

¿Mejora la predicción de un Mundial cuando el modelo conoce más ediciones anteriores?
La respuesta se mide, no se presupone. El resultado esperado no se fuerza a formar una
curva ascendente.

## Corte temporal

El modo predeterminado es `strict_editions`:

1. Ordena todos los partidos cronológicamente.
2. Para un Mundial objetivo, entrena solo con ediciones terminadas anteriormente.
3. Calcula las variables de cada partido con el estado existente antes de jugarlo.
4. Predice toda la edición con el mismo modelo.
5. Dentro de la edición, los partidos revelados actualizan forma, Elo, descanso,
   enfrentamientos y experiencia usados por partidos posteriores.
6. Al terminar, incorpora la edición a la historia del siguiente Mundial.

`walk_forward` reentrena después de cada partido ya revelado. Es temporalmente válido,
pero más lento y menos sencillo de explicar; por eso no es el modo predeterminado.

## Orientación y marcador

La base histórica colocaba al ganador como primer equipo en numerosos partidos antiguos,
lo que filtraba el objetivo. Quinielator ignora esa orientación: si juega un anfitrión lo
coloca primero; en sede neutral ordena por código normalizado. Esta regla usa únicamente
información prepartido.

Se mantienen cuatro cantidades distintas:

- goles al terminar 90 minutos;
- goles totales después de prórroga;
- penaltis de cada equipo;
- selección que avanzó.

El objetivo 1X2 y el marcador predicho siempre corresponden a 90 minutos. Los penaltis
nunca se suman al marcador.

Los signos usan la convención de quiniela: `1` gana el primer equipo mostrado, `X` es
empate a los 90 minutos y `2` gana el segundo. El marcador modal y el signo se calculan
por separado desde la matriz probabilística. El signo agrega todos los marcadores de su
clase; por eso puede no coincidir con la clase del único marcador más probable.

## Variables

El dataset contiene 68 variables numéricas entregadas al modelo:

- Elo, diferencia y expectativa Elo;
- ranking FIFA, diferencia, disponibilidad, antigüedad y obsolescencia;
- diferencia relativa de puntos FIFA;
- anfitrión, fase, año y confederaciones codificadas one-hot;
- partidos, tasas históricas y experiencia en eliminatorias;
- forma reciente ponderada;
- goles recientes a favor/en contra y fuerza Elo de rivales;
- partidos del Mundial actual y días de descanso;
- enfrentamientos directos;
- diferencias entre ambos equipos.

Los puntos FIFA brutos se conservan para auditoría, pero se excluyen del modelo porque
FIFA cambió su escala varias veces. Los códigos numéricos de confederación también se
conservan, pero no se usan: CONMEBOL no es «mayor» que UEFA; el modelo recibe indicadores
nominales independientes. Antes de 1993 no se inventa ranking. Una publicación con más
de 365 días se considera obsoleta; por eso la evaluación 2026 no utiliza como vigente el
último dato abierto de 2024.

## Elo

Cada equipo empieza en 1500. La expectativa del primer equipo es:

```text
E = 1 / (1 + 10 ^ ((Elo_rival - Elo_equipo) / 400))
```

Después del partido se aplica `nuevo = anterior + K × (resultado - expectativa)`, con
`K = 30` por defecto. La actualización ocurre después de crear las variables del partido.

## Modelos

- `BaselineModel`: promedios de goles históricos, ajustados por diferencia Elo.
- `SklearnPoissonModel`: dos pipelines con imputación, escalado y `PoissonRegressor`.
- `SklearnOutcomeModel`: pipeline de regresión logística multiclase.
- `KerasMatchModel`: MLP 32→16, dropout 0,20, dos salidas softplus, pérdida Poisson,
  validación temporal, early stopping y `shuffle=False`.
- `EnsembleModel`: mezcla Poisson/Keras. Prueba pesos 0, 0,25, 0,50, 0,75 y 1 en el
  último 20% del entrenamiento; nunca selecciona el peso con el Mundial objetivo.

Con menos de 200 partidos históricos, los modelos complejos usan el baseline y lo
declaran como `baseline_fallback`.

La comparación, justificación, hiperparámetros y limitaciones de cada alternativa están
en [MODELOS.md](MODELOS.md). Keras es un candidato experimental, no una suposición de
superioridad. Para 2026, la validación previa del ensamble eligió 100% Poisson.

## Probabilidades y métricas

Las intensidades esperadas se convierten en una matriz de marcadores Poisson de 0 al
máximo configurable. La cola superior se acumula en la última celda. De la matriz salen
las probabilidades local/empate/visitante y el marcador modal.

Se informan accuracy 1X2, marcador exacto, selección que avanzó, MAE de goles, log loss,
Brier, cobertura FIFA, tamaño de entrenamiento y media acumulada. Accuracy mide la clase;
log loss y Brier evalúan la calidad probabilística y deben minimizarse.

## Análisis de variables

`analyze-features` vuelve a entrenar el modelo únicamente con ediciones anteriores al
Mundial solicitado. Después calcula importancia por permutación en el Mundial objetivo:

1. mide el log loss original;
2. desordena una variable entre partidos, sin cambiar el objetivo;
3. vuelve a predecir y registra el aumento de log loss;
4. repite con la semilla configurada;
5. hace lo mismo por familias, permutando juntas las columnas relacionadas.

Un Δ log loss positivo significa que romper la señal empeoró al modelo fuera de muestra.
No prueba causalidad. Variables correlacionadas pueden repartirse el aporte; un valor
negativo solo significa que no se detectó una ayuda estable en esa edición. Este análisis
es posterior y no decide el modelo ni sus hiperparámetros con datos del Mundial objetivo.

Para 2026, las familias con mayor aporte fueron experiencia (37,7%), contexto (21,1%) y
forma/resultados (20,5%). El ranking FIFA representó 0,2% porque no había una publicación
válida bajo la regla de 365 días. El detalle reproducible está en
[features_ensemble_2026.md](resultados-2026/features_ensemble_2026.md).

## Salvaguardas automatizadas

- búsqueda FIFA con `bisect_left`, estrictamente anterior a la fecha;
- imputadores y escaladores dentro del pipeline de entrenamiento;
- actualización de estados posterior a la captura de variables;
- test que modifica un resultado futuro sin alterar variables pasadas;
- test de separación 90 minutos/prórroga/penaltis para 2026;
- test de signos 1/X/2 y coloreado del reporte completo;
- test de corte temporal del análisis por permutación;
- serialización y carga de modelos;
- seeds configuradas para Python, NumPy, scikit-learn y TensorFlow.
