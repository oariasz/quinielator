# Importancia de variables — Mundial 2026

Modelo `ensemble` entrenado con 964 partidos anteriores y explicado sobre 104 partidos. Log loss base: **1.0031**.

Peso elegido sin mirar el Mundial objetivo: Poisson **100%** y Keras **0%**.

La importancia es el aumento promedio del log loss al permutar la señal. Un valor positivo indica que el modelo perdía calidad sin esa información. Es una asociación post-hoc, no una relación causal, y no se usó para seleccionar el modelo con el Mundial objetivo.

## Familias de variables

| Pos. | Familia | Fuente | Variables | Δ log loss | Aporte positivo | Interpretación |
|---:|---|---|---:|---:|---:|---|
| 1 | Experiencia | Resultados mundialistas | 9 | 0.0403 ± 0.0087 | 37.7% | Ayuda fuera de muestra |
| 2 | Contexto del partido | Resultados mundialistas | 18 | 0.0226 ± 0.0132 | 21.1% | Ayuda fuera de muestra |
| 3 | Forma y resultados | Resultados mundialistas | 12 | 0.0219 ± 0.0111 | 20.5% | Ayuda fuera de muestra |
| 4 | Goles recientes | Resultados mundialistas | 6 | 0.0116 ± 0.0063 | 10.8% | Ayuda fuera de muestra |
| 5 | Descanso | Resultados mundialistas | 3 | 0.0076 ± 0.0022 | 7.1% | Ayuda fuera de muestra |
| 6 | Elo | Resultados mundialistas | 7 | 0.0028 ± 0.0026 | 2.6% | Ayuda fuera de muestra |
| 7 | Ranking FIFA | Dato-Futbol / FIFA | 11 | 0.0002 ± 0.0004 | 0.2% | Ayuda fuera de muestra |
| 8 | Enfrentamientos directos | Resultados mundialistas | 2 | -0.0022 ± 0.0018 | 0.0% | Sin aporte estable detectado |

## Variables individuales más influyentes

| Pos. | Variable | Familia | Δ log loss | Aporte positivo |
|---:|---|---|---:|---:|
| 1 | `away_confederation_uefa` | Contexto del partido | 0.0093 ± 0.0046 | 9.2% |
| 2 | `away_matches` | Experiencia | 0.0074 ± 0.0032 | 7.3% |
| 3 | `home_is_host` | Contexto del partido | 0.0073 ± 0.0069 | 7.2% |
| 4 | `away_confederation_concacaf` | Contexto del partido | 0.0070 ± 0.0035 | 6.9% |
| 5 | `away_confederation_afc` | Contexto del partido | 0.0062 ± 0.0018 | 6.1% |
| 6 | `away_knockout_experience` | Experiencia | 0.0057 ± 0.0030 | 5.6% |
| 7 | `difference_win_rate` | Forma y resultados | 0.0052 ± 0.0014 | 5.1% |
| 8 | `home_win_rate` | Forma y resultados | 0.0042 ± 0.0017 | 4.1% |
| 9 | `difference_matches` | Experiencia | 0.0040 ± 0.0035 | 3.9% |
| 10 | `difference_loss_rate` | Forma y resultados | 0.0039 ± 0.0017 | 3.8% |
| 11 | `home_rest_days` | Descanso | 0.0037 ± 0.0013 | 3.7% |
| 12 | `difference_knockout_experience` | Experiencia | 0.0037 ± 0.0021 | 3.6% |
| 13 | `home_confederation_afc` | Contexto del partido | 0.0030 ± 0.0068 | 2.9% |
| 14 | `away_recent_goals_for` | Goles recientes | 0.0022 ± 0.0013 | 2.2% |
| 15 | `away_loss_rate` | Forma y resultados | 0.0021 ± 0.0015 | 2.1% |
| 16 | `home_confederation_conmebol` | Contexto del partido | 0.0021 ± 0.0041 | 2.1% |
| 17 | `difference_recent_goals_against` | Goles recientes | 0.0020 ± 0.0016 | 2.0% |
| 18 | `away_elo` | Elo | 0.0018 ± 0.0011 | 1.8% |
| 19 | `home_draw_rate` | Forma y resultados | 0.0018 ± 0.0042 | 1.8% |
| 20 | `away_recent_form` | Forma y resultados | 0.0017 ± 0.0018 | 1.7% |

## Lectura responsable

- Variables correlacionadas pueden repartirse la importancia y parecer pequeñas por separado.
- Un valor negativo no prueba que la variable sea dañina; indica que no mostró aporte estable en esta edición y con estas permutaciones.
- La familia Ranking FIFA puede quedar en cero cuando no existe una publicación suficientemente reciente, como ocurre en 2026 con la copia abierta utilizada.
- La importancia describe el modelo y dataset evaluados, no una ley general del fútbol.
