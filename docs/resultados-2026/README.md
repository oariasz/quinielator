# Mundial 2026: resultados del experimento

Este directorio contiene los artefactos didácticos versionados del modelo `ensemble`.
Todas las predicciones y resultados se evalúan a 90 minutos; prórroga y penaltis solo se
anotan como contexto posterior.

## Tabla de los 104 partidos

- [HTML interactuable y a color](mundial_2026_ensemble.html): verde = signo acertado,
  rojo = fallo y 🎯 = signo más marcador exacto.
- [Tabla Markdown](mundial_2026_ensemble.md).
- [CSV completo](mundial_2026_ensemble.csv).
- [CSV de indicadores por fase](mundial_2026_ensemble_por_fase.csv).
- [Resumen SVG](mundial_2026_ensemble_resumen.svg).

Resultado: 47/104 signos (45,2%), 11/104 marcadores exactos (10,6%) y cuatro tandas
posteriores ignoradas correctamente para 1/X/2.

El HTML, Markdown y SVG incluyen porcentajes por fase. El CSV separado conserva para
cada ronda el número de partidos, aciertos de signo, marcadores exactos y ambos
porcentajes.

## Feature analysis

- [Informe explicado](features_ensemble_2026.md).
- [Gráfico SVG](features_ensemble_2026.svg).
- [CSV por variable](features_ensemble_2026.csv).
- [CSV por familia](features_ensemble_2026_groups.csv).

La importancia se mide por permutación con cinco repeticiones: cuánto aumenta el log loss
al romper una señal. Describe el comportamiento predictivo del modelo en esta edición y
no establece causalidad.

## Regenerar

```bash
python -m quinielator report --model ensemble --world-cup 2026 --publish-docs
python -m quinielator analyze-features --model ensemble --world-cup 2026 --repeats 5 --publish-docs
```

Consulta [modelos y motivos](../MODELOS.md), [metodología](../METODOLOGIA.md),
[resultados](../RESULTADOS.md) y [fuentes de datos](../FUENTES_DE_DATOS.md).
