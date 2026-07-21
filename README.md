# Quinielator

Laboratorio local y didáctico para predecir partidos de los Mundiales masculinos con
validación temporal. Reúne resultados de 1930 a 2026, ranking FIFA cuando es válido,
rating Elo y forma previa; compara un baseline, dos modelos de scikit-learn, una red
TensorFlow/Keras y un ensamble.

El proyecto está implementado con clases, se ejecuta por terminal y no usa Docker,
servidores web ni servicios cloud.

## Resultado real del experimento

Se evaluaron 967 partidos de los Mundiales 1958–2026. Para cada edición, el modelo
solamente aprendió de ediciones anteriores (`strict_editions`).

| Modelo | Accuracy 1X2 | Marcador exacto | MAE goles | Log loss | Brier |
|---|---:|---:|---:|---:|---:|
| Ensemble Poisson + Keras | **51,9%** | **13,2%** | **0,974** | **1,020** | **0,609** |
| Baseline Elo | 50,3% | 12,2% | 1,028 | 1,031 | 0,617 |
| Poisson (scikit-learn) | 50,1% | 12,1% | 0,990 | 1,035 | 0,619 |
| Keras | 49,8% | 10,1% | 1,131 | 1,126 | 0,652 |
| Regresión logística | 42,8% | 10,7% | 1,088 | 1,243 | 0,721 |

En 2026 el ensamble acertó el 56,7% de los resultados 1X2 y la media acumulada pasó
de 51,3% al terminar 2022 a 51,9%. La curva por edición no es monótona: disponer de
más historia no garantiza mejorar en cada Mundial. Ese resultado honesto es parte del
aprendizaje del proyecto.

## Instalación rápida

Requiere Python 3.11. Python 3.12 también es compatible, pero se recomienda reproducir
el entorno probado.

```bash
git clone https://github.com/oariasz/quinielator.git
cd quinielator
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,keras]"
```

Descarga, valida y prepara los datos:

```bash
python -m quinielator data download
python -m quinielator data validate
python -m quinielator data prepare
```

Ejecuta el backtest del mejor modelo y abre la demostración:

```bash
python -m quinielator evaluate --model ensemble
python -m quinielator report --model ensemble
python -m quinielator interactive
```

La primera evaluación con TensorFlow puede tardar algunos minutos. Después, las
consultas por Mundial reutilizan el reporte local.

## Demostración interactiva

La función pública es:

```python
from quinielator import print_predictions

print_predictions(stage="final")
print_predictions(stage="semifinal", start_year=2014)
print_predictions(stage="cuartos", start_year=2022, end_year=2026)
```

Primero imprime la predicción y sus probabilidades. Al pulsar Enter revela el resultado
real; otro Enter avanza al siguiente partido o Mundial. Escribe `q` para salir. Después
de la fase muestra la precisión calculada con todos los partidos de esa edición.

También funciona desde la terminal:

```bash
python -m quinielator interactive --stage final
python -m quinielator interactive --stage seminfinal
python -m quinielator interactive --stage cuartos
python -m quinielator predict --world-cup 2026 --stage final --model ensemble
```

`seminfinal` se acepta deliberadamente como alias de semifinal. El Mundial de 1950 se
omite al solicitar finales porque no tuvo una final oficial.

## Comandos

```text
python -m quinielator data download [--force]
python -m quinielator data validate
python -m quinielator data prepare
python -m quinielator models
python -m quinielator train --model MODEL
python -m quinielator evaluate --model MODEL [--start-year AÑO] [--end-year AÑO]
python -m quinielator report --model MODEL
python -m quinielator interactive --stage FASE --model MODEL
python -m quinielator predict --world-cup AÑO --stage FASE --model MODEL
```

Modelos: `baseline`, `poisson`, `outcome`, `keras` y `ensemble`. El modo riguroso
predeterminado es `strict_editions`; `--mode walk_forward` permite reentrenar después
de cada resultado ya conocido y es mucho más lento.

## Cómo evita información futura

Los partidos se recorren cronológicamente. Antes de cada uno se capturan las 58
variables disponibles y solo después se actualizan Elo, forma, enfrentamientos y
experiencia. Los pipelines ajustan imputadores y escaladores exclusivamente con el
conjunto de entrenamiento.

- El ranking elegido es la publicación estrictamente anterior al partido.
- Un ranking de más de 365 días se marca como obsoleto y no se utiliza.
- Los penaltis no se suman al marcador de 90 minutos.
- La orientación local/visitante se decide por anfitrión o código, nunca por el ganador.
- Las primeras ediciones usan el baseline hasta alcanzar 200 partidos de historia.
- Una prueba cambia un resultado futuro y verifica que las variables pasadas no cambian.

Consulta [METODOLOGIA.md](docs/METODOLOGIA.md) para el diseño completo.

## Arquitectura con clases

```text
src/quinielator/
├── data/          DataDownloader, DataRepository, fuentes y validación
├── domain/        entidades y fases normalizadas
├── features/      Elo, forma, ranking, H2H y dataset temporal
├── models/        interfaz común y cinco modelos
├── evaluation/    backtesting, métricas y reportes SVG/CSV/JSON/Markdown
├── training/      entrenamiento y serialización
├── presentation/  interacción predicción → pausa → resultado
└── services/      fachada de aplicación
```

La configuración reproducible vive en `config/default.toml`. Los datos descargados,
datasets procesados, modelos y reportes se crean localmente y están excluidos de Git.

## Fuentes

- Resultados 1930–2022: [Fjelstul World Cup Database](https://github.com/jfjelstul/worldcup),
  CC BY-SA 4.0.
- Resultados 2026: [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json),
  datos declarados de dominio público por el repositorio.
- Ranking FIFA histórico: [Dato-Futbol/fifa-ranking](https://github.com/Dato-Futbol/fifa-ranking),
  extraído del sitio FIFA. El repositorio no declara una licencia; por ello el CSV no se
  redistribuye.

Cada descarga genera un archivo `.metadata.json` con URL, fecha, SHA-256, tamaño y nota
de licencia. La fuente de ranking usada llega hasta septiembre de 2024: por la regla de
caducidad, 2026 se evalúa sin fingir que ese ranking seguía vigente.

Más detalle en [FUENTES_DE_DATOS.md](docs/FUENTES_DE_DATOS.md).

## Calidad y reproducción

```bash
ruff check .
ruff format --check .
mypy src/quinielator
pytest
```

El workflow de GitHub Actions ejecuta esas comprobaciones sin descargar datasets ni
entrenar TensorFlow. Los resultados medidos están documentados en
[RESULTADOS.md](docs/RESULTADOS.md) y el tutorial completo en
[MANUAL_DE_USO.md](docs/MANUAL_DE_USO.md).

## Limitaciones

- Un acierto 1X2 cercano al 52% es razonable para este experimento, no una garantía.
- Los datos históricos comparables no incluyen lesiones, alineaciones, cuotas o xG.
- El ranking FIFA no existía antes de 1992 y la copia abierta usada termina en 2024.
- La probabilidad de avanzar aproxima una eliminatoria comparando las probabilidades de
  victoria; no modela por separado prórroga y penaltis.
- Los equipos y resultados de un Mundial futuro nunca se inventan.
- Este software es educativo y no está diseñado para apuestas.

Código bajo licencia MIT. Los datasets conservan sus propios términos.
