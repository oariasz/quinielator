# Manual de uso

Este manual asume conocimientos básicos de terminal y Python. Todo queda en tu equipo;
Internet solo se necesita al descargar o actualizar las fuentes.

## 1. Preparar Python en macOS

Comprueba que tienes Python 3.11:

```bash
python3.11 --version
```

Clona el proyecto, crea un entorno aislado y actívalo:

```bash
git clone https://github.com/oariasz/quinielator.git
cd quinielator
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,keras]"
```

`.[dev,keras]` instala scikit-learn, TensorFlow/Keras y las herramientas de calidad. Si
solo quieres probar Poisson y el baseline, `pip install -e ".[dev]"` evita TensorFlow.

Cada vez que abras una nueva terminal, entra al directorio y reactiva el entorno:

```bash
cd quinielator
source .venv/bin/activate
```

## 2. Descargar los datos

```bash
python -m quinielator data download
```

Las fuentes se guardan en `data/raw/`. Si ya existen, no se descargan de nuevo. Para
actualizarlas deliberadamente:

```bash
python -m quinielator data download --force
```

Cada archivo incluye metadatos de procedencia y SHA-256. Los datos grandes no forman
parte del repositorio Git.

## 3. Validar y preparar

```bash
python -m quinielator data validate
python -m quinielator data prepare
```

La validación detecta duplicados, fechas o marcadores inválidos, equipos repetidos y
errores de penaltis. Su reporte queda en `data/processed/validation_report.json`.

La preparación crea:

- `data/processed/matches.csv`: partidos normalizados;
- `data/processed/rankings.csv`: publicaciones FIFA normalizadas;
- `data/processed/features.csv`: 58 variables calculadas antes de cada partido.

Con las fuentes actuales deben aparecer 1.068 partidos y 23 ediciones, de 1930 a 2026.

## 4. Conocer y entrenar modelos

```bash
python -m quinielator models
python -m quinielator train --model poisson
python -m quinielator train --model ensemble
```

`train` ajusta un modelo con toda la historia y lo guarda en `models/`. Sirve para
conservar un artefacto final; la demostración histórica usa `evaluate`, que entrena de
nuevo en cada corte temporal para simular lo que se sabía entonces.

Modelos disponibles:

- `baseline`: medias históricas ajustadas por Elo;
- `poisson`: dos regresores de Poisson de scikit-learn;
- `outcome`: regresión logística para H/D/A;
- `keras`: red pequeña que estima goles esperados;
- `ensemble`: combina Poisson y Keras con un peso elegido en un bloque de validación
  anterior al Mundial evaluado.

## 5. Evaluar sin mirar el futuro

Para el modelo recomendado:

```bash
python -m quinielator evaluate --model ensemble
```

El modo predeterminado `strict_editions` entrena cada edición con Mundiales anteriores.
Los partidos ya disputados dentro de la edición sí modifican sus variables de forma,
pero el modelo no se reajusta hasta la siguiente edición.

Puedes limitar el experimento o usar el modo más costoso por partido:

```bash
python -m quinielator evaluate --model poisson --start-year 1998 --end-year 2026
python -m quinielator evaluate --model poisson --mode walk_forward --start-year 2022
```

Las evaluaciones parciales se guardan con un nombre distinto y nunca sobrescriben el
reporte histórico completo.

## 6. Generar y leer reportes

```bash
python -m quinielator report --model ensemble
```

En `reports/` encontrarás:

- `predictions_ensemble.csv` y `.json`: cada predicción y resultado real;
- `metrics_ensemble.csv`: métricas por Mundial;
- `accuracy_ensemble.svg`: accuracy por edición y media acumulada;
- `report_ensemble.md`: resumen legible;
- `metrics_by_phase_ensemble.csv`: rendimiento por fase;
- `confusion_ensemble.csv`: matriz real frente a predicho;
- `calibration_ensemble.csv`: confianza declarada frente a acierto observado;
- `model_comparison.csv` y `.md`: comparación de todos los modelos evaluados.

Interpretación:

- `outcome_accuracy`: proporción de H/D/A acertados;
- `exact_score_accuracy`: marcadores exactos acertados;
- `goals_mae`: error absoluto promedio por equipo; menor es mejor;
- `log_loss`: castiga probabilidades muy seguras y equivocadas; menor es mejor;
- `brier_score`: error cuadrático de las probabilidades; menor es mejor;
- `calibration_error`: diferencia entre confianza y acierto observado; menor es mejor;
- `accuracy_ci_lower/upper`: intervalo Wilson de 95% para el accuracy;
- `advancing_accuracy`: selección que avanzó acertada;
- `fifa_ranking_coverage`: partidos con ranking válido para ambos equipos.

## 7. Jugar con la predicción interactiva

Finales, semifinales y cuartos:

```bash
python -m quinielator interactive
python -m quinielator interactive --stage semifinal
python -m quinielator interactive --stage cuartos
```

También acepta `semifinals`, `semis`, el error frecuente `seminfinal`, `quarterfinal` y
`quarterfinals`.

Flujo de cada partido:

1. Ves marcador predicho, goles esperados y probabilidades.
2. Pulsas Enter y aparece el resultado real.
3. Ves si acertó 1X2, selección que avanzó y marcador exacto.
4. Pulsas Enter para continuar; escribe `q` en cualquier pausa para salir.
5. Al acabar esa fase, ves las métricas de todos los partidos del Mundial.

Acota los años para una sesión corta:

```bash
python -m quinielator interactive --stage final --start-year 2018 --end-year 2026
```

Consulta sin pausas:

```bash
python -m quinielator predict --world-cup 2026 --stage final --model ensemble
python -m quinielator predict --world-cup 2026 --stage cuartos --model ensemble
```

## 8. Usar `print_predictions` desde Python

Abre `python` con el entorno activo:

```python
from quinielator import print_predictions

print_predictions()
print_predictions(stage="seminfinal", start_year=2014, end_year=2026)
print_predictions(stage="cuartos", model_name="poisson")
```

La firma completa es:

```python
print_predictions(
    stage="final",
    start_year=None,
    end_year=None,
    model_name="ensemble",
    evaluation_mode="strict_editions",
)
```

## 9. Reproducir las comprobaciones

```bash
ruff check .
ruff format --check .
mypy src/quinielator
pytest
```

Las pruebas no descargan Internet. Cubren Elo, ranking estrictamente anterior,
caducidad del ranking, ausencia de fuga temporal, scores y probabilidades, penaltis,
serialización, aliases de fases y la pausa interactiva.

## 10. Errores frecuentes

`Falta data/raw/...`
: Ejecuta `data download`, luego `data validate` y `data prepare`.

`TensorFlow no está instalado`
: Ejecuta `pip install -e ".[keras]"` o usa `--model poisson`.

`No hay evaluación para ...`
: Ejecuta primero `python -m quinielator evaluate --model NOMBRE`.

La descarga falla por red
: Comprueba la conexión y vuelve a ejecutar el mismo comando. Los archivos ya
  descargados permanecen en caché.

No aparece una final en 1950
: Es correcto: esa edición tuvo una liguilla final y no una final oficial.

La precisión baja en un Mundial
: No es un error. Los estilos, formatos y distribuciones cambian; más datos no garantizan
  una mejora monótona.

## 11. Cambiar parámetros

Edita `config/default.toml` para variar la ventana de forma, Elo, máximo de goles,
semilla, mínimo de entrenamiento y épocas de Keras. Después vuelve a ejecutar
`data prepare` si cambias features, y `evaluate` para recalcular resultados.

No edites los CSV procesados manualmente: son artefactos regenerables.
