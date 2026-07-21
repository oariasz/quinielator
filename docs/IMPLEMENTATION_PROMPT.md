# Prompt de implementación autónoma de Quinielator

Quiero que implementes de principio a fin un proyecto didáctico de machine learning en Python llamado “Quinielator”.

Debes trabajar autónomamente hasta tener una implementación funcional, probada, documentada y publicada en GitHub.

## Naturaleza del proyecto

Este es un proyecto personal, didáctico y exclusivamente local. No uses la guía de proyectos de Estratek. No añadas Docker, FastAPI, frontend, despliegue, infraestructura cloud ni otras capas que no ayuden directamente al experimento de machine learning.

## Ubicación y repositorio

Trabaja en este directorio local:

`/Users/oariasz/dev/Python/quinielator`

Crea o reutiliza este repositorio:

`https://github.com/oariasz/quinielator`

Reglas:

- Si el directorio ya contiene archivos, inspecciónalos antes de modificar nada.
- Conserva cualquier trabajo existente que sea útil.
- No borres ni sobrescribas cambios existentes sin comprobarlos.
- Si el repositorio GitHub no existe, créalo en la cuenta `oariasz`.
- Usa visibilidad pública.
- Configura `main` como rama principal.
- Incluye una licencia MIT para el código.
- Realiza commits pequeños, claros y descriptivos.
- Publica el proyecto terminado en GitHub.
- No subas datasets grandes, modelos entrenados pesados, secretos ni archivos temporales.
- Incluye fixtures pequeños para que las pruebas puedan ejecutarse sin descargar todos los datos.

El software debe ejecutarse localmente. No quiero desplegar una aplicación web ni depender de servicios cloud.

Internet solamente se utilizará para descargar o actualizar los datos. Después de descargarlos, el entrenamiento, evaluación y uso interactivo deben poder ejecutarse localmente desde la caché.

## Objetivo del proyecto

El programa debe recopilar, normalizar y almacenar localmente:

1. Los resultados disponibles de todos los Mundiales masculinos de fútbol.
2. Las fechas de los partidos.
3. Las selecciones participantes.
4. Las fases de cada partido.
5. Los marcadores.
6. Los resultados al final de los 90 minutos.
7. Los resultados después de prórroga.
8. Las tandas de penaltis, manteniéndolas separadas del marcador.
9. El ranking FIFA disponible para cada selección y fecha.
10. Otros datos históricos consistentes que puedan utilizarse para predecir partidos a través de distintas épocas.

Después debe crear, entrenar y comparar modelos de machine learning usando:

- scikit-learn.
- TensorFlow.
- Keras.
- Modelos estadísticos apropiados para marcadores de fútbol.
- Un ensemble, si mejora realmente los resultados de validación.

El proyecto debe poder predecir:

- Marcador más probable.
- Goles esperados de cada equipo.
- Probabilidad de victoria de cada selección.
- Probabilidad de empate.
- Resultado 1X2.
- Ganador o selección que avanza en una eliminatoria.
- Partidos de la final.
- Partidos de las semifinales.
- Partidos de los cuartos de final.
- Todos los partidos de un Mundial para calcular la precisión completa de esa edición.

## Propósito didáctico

El propósito es demostrar cómo cambia la capacidad predictiva al acumular más información histórica.

Para evaluar un Mundial determinado, el modelo solamente puede aprender de datos conocidos previamente.

Ejemplo:

- Para evaluar 1998, entrena inicialmente con los Mundiales anteriores a 1998.
- Dentro de 1998, los partidos ya jugados pueden actualizar la forma y estadísticas de las selecciones antes de predecir partidos posteriores.
- Para evaluar 2002, utiliza los datos acumulados hasta terminar 1998.
- Continúa así cronológicamente.

Quiero visualizar si la precisión mejora a medida que el modelo dispone de más Mundiales históricos.

No manipules los resultados para obtener una curva siempre ascendente. Es posible que la precisión disminuya en algunas ediciones. Debes mostrar honestamente:

- Precisión por Mundial.
- Media acumulada.
- Número de partidos usados para entrenar.
- Cantidad de ediciones conocidas.
- Comparación contra modelos baseline.
- Intervalos de confianza cuando sean razonables.
- Una gráfica de la evolución histórica.

Explica claramente que más datos no garantizan una mejora monótona.

## Regla fundamental contra data leakage

Para predecir un partido solamente pueden utilizarse datos existentes antes de su fecha y hora.

Nunca debes utilizar:

- El resultado del partido que se está prediciendo.
- Estadísticas calculadas con partidos posteriores.
- Rankings publicados después del encuentro.
- El resultado final del Mundial.
- La fase alcanzada por una selección, si todavía no se conocía.
- Transformaciones ajustadas con el conjunto de prueba.
- Información del futuro utilizada para rellenar datos históricos.
- Variables derivadas del resultado objetivo.

Reglas obligatorias:

- No uses una división aleatoria tradicional de train/test.
- Implementa validación temporal expanding-window por Mundial.
- Ajusta imputadores, escaladores, selectores y modelos solamente con datos de entrenamiento.
- Genera las predicciones en orden cronológico.
- Guarda cada predicción antes de revelar el resultado real.
- Añade pruebas automatizadas específicas contra data leakage.
- Incluye una prueba que demuestre que modificar un resultado futuro no cambia una predicción pasada.

Quiero dos comportamientos diferenciados y documentados:

1. `strict_editions`:
   - El modelo se entrena solamente con ediciones anteriores.
   - Los partidos del Mundial actual actualizan features de forma, pero no vuelven a entrenar el modelo.

2. `walk_forward`:
   - Cada resultado revelado puede incorporarse al entrenamiento antes de predecir el siguiente partido.
   - Todo debe ocurrir respetando estrictamente el orden temporal.

El modo por defecto debe ser el que produzca una demostración más rigurosa y comprensible. Justifica la decisión.

## Fuentes de datos

Investiga fuentes actuales, legales, reproducibles y razonablemente confiables.

Prioriza:

1. Fuentes oficiales.
2. Datos abiertos con procedencia verificable.
3. Repositorios mantenidos y auditables.
4. Fuentes alternativas solamente cuando las anteriores no cubran las ediciones históricas.

Implementa adaptadores separados para cada fuente.

Para cada dataset registra:

- Nombre de la fuente.
- URL.
- Fecha de descarga.
- Licencia o términos conocidos.
- Versión.
- Hash del archivo descargado.
- Columnas originales.
- Transformaciones aplicadas.
- Cobertura temporal.
- Limitaciones conocidas.

No bases la solución en scraping frágil si existe una fuente descargable más estable.

Si una fuente deja de funcionar, investiga otra alternativa segura, implementa el adaptador correspondiente, documenta el cambio y continúa autónomamente.

Organiza los datos así:

```text
data/
    raw/
    interim/
    processed/
    fixtures/
```

No subas los datasets grandes al repositorio. Incluye datos de ejemplo pequeños y suficientes para las pruebas.

## Validación de datos

Implementa validaciones para detectar:

- Partidos duplicados.
- Fechas inválidas.
- Marcadores negativos o imposibles.
- Selecciones desconocidas.
- Códigos de país inconsistentes.
- Rankings posteriores al partido.
- Fases incorrectas.
- Mundiales inexistentes.
- Resultados de penaltis mezclados con el marcador.
- Partidos sin equipos.
- Equipos duplicados en un mismo partido.
- Orden cronológico incorrecto.
- Inconsistencias entre ganador y marcador.

Genera un reporte de calidad de los datos.

## Normalización histórica

Normaliza los nombres históricos de las selecciones sin perder los datos originales. Ten en cuenta Alemania y Alemania Occidental, Unión Soviética y sucesoras, Yugoslavia, Checoslovaquia, Zaire y República Democrática del Congo, cambios de códigos FIFA y variantes ortográficas.

No combines automáticamente entidades históricas diferentes sin una regla explícita y documentada. Conserva nombre y código originales, nombre y código normalizados y periodo de validez del mapeo.

## Ranking FIFA y rating Elo

El ranking FIFA no existe para los primeros Mundiales.

Por lo tanto:

- Usa el ranking FIFA más reciente publicado antes de cada partido.
- Nunca uses un ranking posterior.
- Incluye `fifa_ranking_available`.
- Incluye indicadores de valores ausentes.
- No inventes rankings FIFA históricos.
- No rellenes rankings antiguos con datos futuros.
- Implementa un rating Elo cronológico como medida de fuerza disponible para todas las épocas donde existan resultados previos.
- Inicializa y actualiza Elo sin utilizar resultados futuros.
- Documenta la fórmula, parámetros y sensibilidad del Elo.
- Permite configurar la importancia de amistosos, eliminatorias y partidos de Mundial si incorporas partidos internacionales externos.

## Particularidades históricas

El código debe manejar correctamente los cambios de formato de los Mundiales:

- No todas las ediciones tuvieron el mismo número de participantes.
- No todas tuvieron octavos, cuartos o semifinales.
- Algunas utilizaron una segunda fase de grupos.
- El Mundial de 1950 no tuvo una final oficial; no inventes una.
- Algunos partidos se decidieron después de prórroga.
- Algunos se decidieron mediante penaltis, que no forman parte del marcador normal.
- Hubo ediciones canceladas o no disputadas.
- La cantidad de partidos por fase varía.

Si una edición no contiene la fase solicitada, muestra una explicación y continúa sin fallar.

## Features

Implementa como mínimo features calculadas siempre con información previa al partido:

- Ranking FIFA, puntos FIFA y sus diferencias.
- Indicador de ranking FIFA ausente.
- Elo de ambos equipos y diferencia Elo.
- Forma, victorias, empates y derrotas recientes.
- Goles anotados, recibidos y diferencia de goles reciente.
- Promedios móviles configurables y ponderados por recencia.
- Rendimiento histórico en Mundiales y por fase.
- Experiencia acumulada en partidos mundialistas.
- Condición de anfitrión y campo neutral.
- Confederación y enfrentamiento entre confederaciones.
- Días de descanso.
- Fase, año y época futbolística.
- Fuerza de los rivales enfrentados.
- Enfrentamientos directos conocidos previamente.
- Cantidad de partidos conocidos antes de la predicción.
- Tendencias ofensivas y defensivas.
- Rendimiento en el Mundial actual hasta ese momento.
- Fatiga aproximada si los datos lo permiten de forma consistente.

Para variables que no existan en todas las ediciones, añade indicadores de disponibilidad, documenta su cobertura, no uses información futura para imputarlas y evalúa si mejoran la validación temporal.

No utilices como features históricas principales xG, posesión, tiros o pases si no existe cobertura comparable en los Mundiales antiguos.

## Modelos

Define una interfaz común para todos los modelos.

### 1. Baseline ingenuo

Incluye al menos favorito según Elo, favorito según ranking FIFA cuando exista, marcador por promedios históricos y predicción por frecuencia de resultados.

### 2. Modelo estadístico con scikit-learn

Implementa un `PoissonRegressor` para los goles de cada equipo o una solución equivalente correctamente justificada. Convierte los goles esperados en una matriz de probabilidades de marcadores. La cantidad máxima de goles debe ser configurable y debe acumularse correctamente la cola superior.

### 3. Clasificador scikit-learn

Implementa Logistic Regression, HistGradientBoostingClassifier o RandomForestClassifier y elige mediante validación temporal. Usa pipelines para imputación, escalado, transformaciones, entrenamiento y predicción.

### 4. Modelo TensorFlow/Keras

Implementa una red neuronal pequeña y regularizada con salidas para resultado o distribución de goles, early stopping, semillas reproducibles, validación temporal, guardado, carga, historial de entrenamiento y control del sobreajuste. Justifica la arquitectura y no asumas que será superior.

### 5. Ensemble

Implementa un ensemble solo si mejora la validación temporal. Sus pesos deben obtenerse con validación previa, nunca con el Mundial de prueba. Permite ejecutar cada modelo individualmente y guarda pesos y metadatos.

## Predicción de marcadores

Cada predicción debe incluir equipos, fecha, Mundial, fase, goles esperados, marcador más probable, probabilidades 1X2, selección con mayor probabilidad de avanzar, modelo, versión de datos, fecha límite, features principales e incertidumbre.

En eliminatorias diferencia el resultado a 90 minutos, el resultado tras prórroga, el ganador por penaltis y la selección que avanza. No sumes penaltis al marcador normal.

## Métricas

Calcula:

- Accuracy 1X2.
- Accuracy del ganador y de la selección que avanza.
- Accuracy de marcador exacto.
- MAE de goles por equipo y total.
- Log loss.
- Brier score.
- Calibración de probabilidades.
- Matriz de confusión.
- Precisión por fase, Mundial y acumulada.
- Comparación con baselines y entre modelos.
- Cantidad de predicciones evaluadas.
- Cobertura y rendimiento con/sin ranking FIFA.

Cuando una métrica no pueda calcularse, explica el motivo.

## Evaluación walk-forward

Implementa un evaluador temporal que ordene las ediciones, entrene con la historia disponible, prediga cada partido antes de revelar el resultado, guarde la predicción, revele el resultado, actualice el estado histórico, continúe cronológicamente, calcule las métricas de la edición y añada la edición al conjunto histórico.

Si las primeras ediciones no tienen suficientes datos, usa un baseline, indica la limitación y comienza los modelos complejos desde un volumen mínimo configurable.

## Función interactiva obligatoria

Debe existir:

```python
print_predictions(
    stage: str = "final",
    start_year: int | None = None,
    end_year: int | None = None,
    model_name: str = "ensemble",
    evaluation_mode: str = "strict_editions",
) -> None
```

La función será una fachada que delegue en clases como `PredictionService` e `InteractivePredictionPrinter` y funcionará también desde CLI.

Entradas aceptadas para `stage`: `final`, `semifinal`, `semifinals`, `semis`, `seminfinal`, `cuartos`, `quarterfinal` y `quarterfinals`. La fase predeterminada es `final`.

## Comportamiento interactivo

Para cada partido de la fase seleccionada:

1. Muestra Mundial, fase y equipos.
2. Muestra primero solo la predicción, marcador, probabilidades y modelo.
3. Espera Enter.
4. Revela el resultado real y los aciertos de 1X2, ganador, equipo que avanzó y marcador exacto.
5. Espera otro Enter para continuar.
6. Permite `q` para salir.

En semifinales presenta dos partidos y en cuartos cuatro cuando ese formato exista. Si no existe la fase, explica por qué y continúa. Al finalizar la fase, muestra métricas de todos los partidos del Mundial, no solo los interactivos.

La interacción debe poder probarse inyectando `input_fn`, `output_fn` o una interfaz de consola.

## CLI

Implementa comandos equivalentes a:

```bash
python -m quinielator data download
python -m quinielator data validate
python -m quinielator data prepare
python -m quinielator train
python -m quinielator evaluate
python -m quinielator report
python -m quinielator interactive
python -m quinielator interactive --stage final
python -m quinielator interactive --stage semifinal
python -m quinielator interactive --stage cuartos
python -m quinielator interactive --model keras
python -m quinielator predict --world-cup YEAR --stage final
python -m quinielator predict --world-cup YEAR --team-a TEAM --team-b TEAM
```

Permite configurar año inicial/final, modelo, modo de evaluación, semilla, directorios, fecha límite y logging. Si no se conocen participantes futuros, no inventes equipos.

## Arquitectura orientada a objetos obligatoria

La implementación principal debe usar clases. Evita scripts extensos, funciones globales, estado global, clases gigantes, duplicación y sobreingeniería. Usa composición, interfaces o clases abstractas, inyección de dependencias, type hints, `dataclasses`, docstrings en español y métodos pequeños.

Clases o abstracciones equivalentes:

- Dominio: `Team`, `Match`, `Tournament`, `FifaRankingEntry`, `EloRating`, `MatchFeatures`, `MatchPrediction`, `EvaluationMetrics`.
- Datos: `DataSource`, `WorldCupResultsSource`, `FifaRankingsSource`, `DataRepository`, `DataDownloader`, `DataValidator`, `TeamNameNormalizer`.
- Features: `EloRatingCalculator`, `TeamFormCalculator`, `HeadToHeadCalculator`, `FeatureEngineer`, `TemporalDatasetBuilder`.
- Modelos: `PredictionModel`, `BaselineModel`, `SklearnPoissonModel`, `SklearnOutcomeModel`, `KerasMatchModel`, `EnsembleModel`, `ModelRegistry`, `ModelSerializer`.
- Entrenamiento: `ModelTrainer`, `WalkForwardTrainer`, `WorldCupEvaluator`, `MetricsCalculator`, `CalibrationEvaluator`.
- Aplicación: `PredictionService`, `InteractivePredictionPrinter`, `ReportGenerator`, `ApplicationConfig`.

Puedes ajustar nombres, pero conserva la separación de responsabilidades.

## Estructura sugerida

```text
src/quinielator/
    __init__.py
    __main__.py
    cli.py
    config.py
    domain/
    data/
    features/
    models/
    training/
    evaluation/
    services/
    presentation/
tests/
    unit/
    integration/
    fixtures/
config/
data/
models/
reports/
notebooks/
docs/
```

Los notebooks son opcionales y deben consumir las clases del paquete.

## Configuración y reproducibilidad

Usa `pyproject.toml`. Selecciona una versión de Python realmente compatible con scikit-learn y TensorFlow/Keras, especialmente en macOS/Apple Silicon.

Incluye configuración TOML o YAML, semillas para Python/NumPy/scikit-learn/TensorFlow, logging, caché local, guardado de modelos, metadatos, versión/hash de datos, `as_of_date`, configuración de features/Elo/validación y `.gitignore`. No incluyas secretos.

## Informes

Genera en `reports/` predicciones CSV/JSON, métricas por Mundial/fase/modelo, comparación con baselines, evolución y media acumulada, cantidad de datos, matriz de confusión, calibración, comparación de modelos, importancia de features, cobertura FIFA, calidad de datos e informe Markdown. Las gráficas deben estar rotuladas en español.

## Calidad

Configura pytest, pruebas unitarias e integración, fixtures, Ruff, formateo, mypy, cobertura y GitHub Actions. Añade pruebas de normalización, Elo, forma, ranking temporal, penaltis, features, probabilidades, métricas, fases históricas, 1950, alias `seminfinal`, salida `q`, serialización, data leakage y reproducibilidad.

## README y manual

Escribe un README principalmente en español con objetivo, arquitectura, instalación, comandos, modelos, Elo, walk-forward, data leakage, métricas, limitaciones, resultados reales, fuentes/licencias y extensión del sistema.

Crea `docs/MANUAL_DE_USO.md` para una persona con Python básico. Explica clonación, versión de Python, entorno virtual en macOS, instalación, descarga/validación/preparación, entrenamiento, evaluación, reportes, `print_predictions`, fases, modelos, interpretación, ubicación de artefactos, reproducción, actualización y errores comunes.

Incluye y verifica comandos copiables:

```bash
git clone https://github.com/oariasz/quinielator.git
cd quinielator
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,keras]"
python -m quinielator data download
python -m quinielator data validate
python -m quinielator data prepare
python -m quinielator train
python -m quinielator evaluate
python -m quinielator report
python -m quinielator interactive
python -m quinielator interactive --stage semifinal
python -m quinielator interactive --stage cuartos
```

No documentes comandos que no hayas probado.

## Demostración final

Ejecuta descarga/carga de datos reales, validación, preparación, entrenamiento, evaluación walk-forward, reportes y demostraciones de final, semifinales y cuartos. Presenta métricas reales y comparación con baselines. Si TensorFlow tarda demasiado, usa una configuración de demostración razonable pero deja la completa disponible.

## GitHub Actions

Configura un workflow sencillo que instale el paquete y ejecute Ruff, mypy y tests con fixtures. No descargues datasets enormes ni entrenes redes completas en CI.

## Mandato de autonomía

No te limites a plan, arquitectura, carpetas, pseudocódigo o MVP incompleto. No dejes métodos con `pass`, tareas esenciales como TODO ni simules comandos.

Debes inspeccionar, investigar, implementar, descargar datos, validar, crear features, modelar, entrenar, evaluar, implementar la interacción, probar, documentar, versionar y publicar.

Toma autónomamente las decisiones rutinarias. Si algo falla, investiga una alternativa segura y continúa. Pregunta solo ante credenciales, permisos externos, acciones destructivas no autorizadas o decisiones de producto que cambien materialmente el alcance.

## Criterio de terminado

No termines hasta que existan datos reales cacheables, arquitectura con clases, validación temporal, baseline, scikit-learn, TensorFlow/Keras, ensemble o una explicación métrica para no usarlo, evaluación walk-forward, `print_predictions`, fases históricas, métricas completas, pruebas/lint/tipado verdes, reportes reales, manual verificado y publicación en GitHub.

## Entrega y enseñanza final

Al finalizar, entrega URL del repositorio, resumen, arquitectura, fuentes, modelos, métricas, comparación, limitaciones, comandos exactos, ejemplos por fase, ubicación de reportes, archivos clave y un tutorial de primera ejecución. Guíame paso a paso después de haber terminado y probado el proyecto.
