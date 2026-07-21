# Fuentes de datos

Esta página documenta todas las entradas externas usadas en la ejecución de referencia.
Quinielator no consulta una API durante el entrenamiento: descarga archivos, guarda su
huella criptográfica y trabaja después con copias locales reproducibles.

## Inventario exacto de la ejecución

| Archivo local | Origen | Descarga UTC | Bytes | SHA-256 | Términos |
|---|---|---|---:|---|---|
| `world_cup_matches.csv` | [Fjelstul `matches.csv`](https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/matches.csv) | 2026-07-20 23:09:00 | 298.930 | `037f718764758e7dda3a2918276f85fa8389af1d50999d2a480c8a6a49408875` | CC BY-SA 4.0 |
| `world_cup_goals.csv` | [Fjelstul `goals.csv`](https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/goals.csv) | 2026-07-20 23:09:01 | 734.494 | `4fbe990b7f42bb8af00d1587af09359ecfdb4ee4101bb44fda0eb336fc1e5be3` | CC BY-SA 4.0 |
| `world_cup_teams.csv` | [Fjelstul `teams.csv`](https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/teams.csv) | 2026-07-20 23:09:02 | 24.772 | `a2d5ef96fade3bee9ffd22a28cda6378c6b9fb67f8e99514aa58ec0854d26a66` | CC BY-SA 4.0 |
| `world_cup_tournaments.csv` | [Fjelstul `tournaments.csv`](https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/tournaments.csv) | 2026-07-20 23:09:02 | 3.287 | `906b122d80aae5302486bafa9dfd925378503a2a11de6f5db39d6c4879040ce0` | CC BY-SA 4.0 |
| `world_cup_2026.json` | [openfootball `2026/worldcup.json`](https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json) | 2026-07-20 23:49:50 | 42.736 | `7015c672bfead58df5297eb2f2f687594bb7c463508c39076010706236b4e6d6` | dominio público / CC0 |
| `fifa_rankings.csv` | [Dato-Futbol `ranking_fifa_historical.csv`](https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/ranking_fifa_historical.csv) | 2026-07-20 23:09:04 | 2.718.603 | `d4f4d8d3db8e7560823b31b8db7848e1b85fcd1e1353de1d7e1fbedc6d47226e` | sin licencia declarada |

Las fechas y hashes anteriores proceden de los seis `*.metadata.json` creados junto a
las descargas. Los datos crudos no se versionan en este repositorio.

## Resultados históricos: 1930–2022

La [Fjelstul World Cup Database](https://github.com/jfjelstul/worldcup), de Joshua C.
Fjelstul, aporta 964 partidos de 22 Mundiales masculinos. Su estructura, documentación y
datos se publican con licencia CC BY-SA 4.0. El proyecto usa:

- `matches.csv`: identificador, edición, fecha/hora, fase, equipos, marcador total,
  prórroga, tanda y penaltis;
- `goals.csv`: `match_id`, minuto reglamentario y lado del equipo para reconstruir el
  marcador exacto a 90 minutos;
- `teams.csv`: código de selección y confederación;
- `tournaments.csv`: edición y país o países anfitriones.

Se filtran exclusivamente torneos `FIFA Men's World Cup`. Las modificaciones hechas por
Quinielator son: normalización de nombres/códigos, reconstrucción del 90′, separación de
prórroga y tanda, incorporación de anfitrión/confederación y reorientación de los equipos
mediante una regla disponible antes del partido. Esta última evita usar el orden de la
fuente cuando ese orden estaba relacionado con el ganador.

## Resultados 2026

El suplemento procede de
[openfootball/worldcup.json](https://github.com/openfootball/worldcup.json), generado a
partir del repositorio público [openfootball/worldcup](https://github.com/openfootball/worldcup).
Su README lo describe como un esfuerzo abierto de dominio público y el repositorio fuente
declara licencia CC0 1.0.

Se usan `date`, `time`, `round`, `team1`, `team2` y los componentes de `score`:

- `ft`: marcador al finalizar 90 minutos, objetivo de 1/X/2 y marcador exacto;
- `et`: marcador después de la prórroga, solo para saber quién avanzó;
- `p`: tanda de penaltis, guardada aparte y nunca sumada al 90′.

La copia usada contiene los 104 partidos, incluida la ronda de 32. Si la fuente histórica
llegara a incluir 2026, el repositorio elimina el suplemento duplicado. Los nombres,
códigos, confederaciones y anfitriones de este adaptador están declarados explícitamente
en código y son auditables.

## Ranking FIFA

La fuente es [Dato-Futbol/fifa-ranking](https://github.com/Dato-Futbol/fifa-ranking).
Según su README, `ranking_fifa_historical.csv` fue extraído del sitio oficial de FIFA y
cubre diciembre de 1992 a septiembre de 2024. La copia procesada contiene 67.883 filas.
Se usan fecha de publicación, selección, abreviatura, posición y puntos totales.

Para cada partido se busca con `bisect_left` la última publicación **estrictamente
anterior**. Si tiene más de 365 días, posición y puntos pasan a ausentes y se conserva la
antigüedad para auditoría. No se inventan rankings anteriores a 1992 ni se rellena el
pasado con una publicación futura. Por ello el Mundial 2026 tiene 0% de cobertura FIFA
válida con esta copia.

El repositorio fuente no declara licencia. En consecuencia, Quinielator documenta URL y
huella, pero no redistribuye ese CSV.

## De la fuente al dataset de modelos

`data prepare` produce 1.068 partidos de 23 ediciones y una fila de variables por
partido. Cada fila se captura antes de actualizar Elo, forma, descanso, experiencia o
enfrentamientos con el resultado de ese mismo juego. Las 68 variables entregadas al
modelo proceden de:

- resultados anteriores: Elo, forma, goles recientes, experiencia, descanso y H2H;
- contexto conocido: año, fase, anfitrión y confederaciones one-hot;
- ranking: posición, diferencias, disponibilidad, antigüedad y obsolescencia.

Los puntos FIFA brutos y los códigos numéricos de confederación quedan en el CSV
procesado para auditoría, pero se excluyen del aprendizaje. Los puntos cambiaron de
escala históricamente y una confederación es una categoría, no una magnitud ordinal.

## Trazabilidad y actualización

```bash
python -m quinielator data download
python -m quinielator data validate
python -m quinielator data prepare
```

`data download --force` actualiza deliberadamente las copias y, por tanto, puede cambiar
hashes, filas y resultados. La validación detecta duplicados, marcadores incoherentes,
fechas inválidas y problemas de tandas. Los artefactos bajo `data/`, `models/` y
`reports/` son locales; solo los reportes didácticos seleccionados de 2026 se publican en
`docs/resultados-2026/`.

## Limitaciones de cobertura

- No hay una fuente comparable para todos los Mundiales de lesiones, alineaciones, xG,
  posesión, tiros o cuotas; por eso no entran en el experimento histórico.
- Fjelstul se apoya principalmente en páginas archivadas de Wikipedia y contrasta parte
  de la información con informes FIFA; no es una API oficial de resultados.
- Openfootball se actualiza de forma colaborativa, no como feed oficial en tiempo real.
- La cobertura FIFA acaba en 2024 y su ausencia en 2026 debe interpretarse literalmente.
- Los hashes identifican esta ejecución, no garantizan que el proveedor nunca corrija sus
  archivos.
