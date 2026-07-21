# Fuentes de datos

## Resultados 1930–2022

- Fuente: [Fjelstul World Cup Database](https://github.com/jfjelstul/worldcup).
- Autor: Joshua C. Fjelstul, Ph.D.
- Licencia: CC BY-SA 4.0.
- Archivos usados: `matches.csv`, `goals.csv`, `teams.csv`, `tournaments.csv`.
- Cobertura usada: 964 partidos de 22 Mundiales masculinos.

Transformaciones principales: filtrar torneos masculinos, reconstruir goles a 90
minutos desde eventos, separar prórroga/penaltis, normalizar códigos, añadir
confederación/anfitrión y reemplazar la orientación potencialmente filtrada por una regla
prepartido.

## Resultados 2026

- Fuente: [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json).
- Archivo: `2026/worldcup.json`.
- Términos declarados por el repositorio: dominio público.
- Cobertura: 104 partidos, incluida la ronda de 32.

El adaptador separa `ft` (90 minutos), `et` (tras prórroga) y `p` (penaltis), y aplica la
misma orientación prepartido del histórico. Este suplemento se excluye automáticamente si
la fuente histórica llega a incorporar 2026 en una actualización futura.

## Ranking FIFA

- Fuente: [Dato-Futbol/fifa-ranking](https://github.com/Dato-Futbol/fifa-ranking).
- Procedencia declarada: sitio oficial FIFA.
- Cobertura de la copia usada: diciembre de 1992 a septiembre de 2024.
- Filas procesadas: 67.883.
- Licencia: no declarada en el repositorio fuente.

Por la falta de una licencia explícita, el CSV descargado no se redistribuye. Se usa la
última publicación estrictamente anterior al partido. Si tiene más de 365 días, se marca
obsoleta y el modelo recibe el mismo tratamiento que un ranking ausente. No se inventan
rankings anteriores a 1992 ni se rellenan con el futuro.

## Trazabilidad local

`python -m quinielator data download` crea junto a cada archivo un
`NOMBRE.metadata.json` con:

- nombre y URL;
- fecha/hora UTC de descarga;
- SHA-256;
- tamaño en bytes;
- nota de licencia o términos conocidos.

Los archivos `data/raw/`, `data/processed/`, `models/` y los reportes generados están
excluidos de Git. Esto respeta tamaño, reproducibilidad y términos de las fuentes. Para
auditar la transformación, usa:

```bash
python -m quinielator data validate
python -m quinielator data prepare
```

## Limitaciones

- La comparación histórica favorece variables disponibles de manera consistente; no se
  incorporan xG, tiros, posesión o alineaciones modernas.
- La escala de puntos FIFA cambió; los puntos brutos se conservan, pero el modelo usa
  rango y diferencia relativa acotada.
- El ranking abierto termina en 2024, por lo que 2026 queda sin cobertura válida bajo la
  regla de 365 días.
- Actualizar fuentes puede cambiar filas y resultados. Los hashes permiten identificar la
  versión exacta usada en una ejecución.
