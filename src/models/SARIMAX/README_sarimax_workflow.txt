README - Flujo SARIMAX

Objetivo general
----------------
Este documento explica el flujo seguido para incorporar SARIMAX al TFM como modelo estadístico comparativo frente a Prophet.

El objetivo fue evaluar si una estructura SARIMAX podía capturar la demanda eléctrica horaria usando variables meteorológicas, calendario y dependencia temporal.

Archivos involucrados
---------------------
Se trabajó con los siguientes scripts:

1. sarimax_evaluacion.py
2. sarimax_walk_forward.py
3. sarimax_walk_forward_festivos.py

Aclaración sobre el script definitivo
-------------------------------------
El script que debe considerarse definitivo para la evaluación walk-forward de SARIMAX es:

sarimax_walk_forward_festivos.py

Motivo:

- sarimax_walk_forward.py permitió construir la primera comparación walk-forward entre SARIMAX base, SARIMAX + lag 168h y baselines.
- sarimax_walk_forward_festivos.py mantiene esa lógica, pero añade un diagnóstico metodológico clave: identifica si el periodo de test o la semana usada por demanda_lag_168h contienen festivos.

Por tanto, sarimax_walk_forward_festivos.py es la versión más completa y metodológicamente más defendible.

Recomendación práctica:

- Mantener sarimax_walk_forward.py como versión preliminar o histórica.
- Usar sarimax_walk_forward_festivos.py como versión definitiva para análisis y resultados.
- Alternativamente, si el equipo prefiere un nombre limpio, copiar el contenido definitivo de sarimax_walk_forward_festivos.py dentro de sarimax_walk_forward.py y dejar constancia del cambio.

1. sarimax_evaluacion.py
------------------------
Función del archivo:

sarimax_evaluacion.py se creó como script de diagnóstico inicial. Su objetivo fue validar que SARIMAX podía entrenarse sobre el dataset, generar predicciones y calcular MAPE en un único corte temporal.

Corte usado:

- 2025-10-03

Horizonte:

- 3 días

Ventana de entrenamiento:

- 365 días

Principales decisiones metodológicas:

- No se forzó la frecuencia horaria con asfreq("h") porque se detectó que podía crear filas artificiales con NaN durante cambios horarios, por ejemplo en el cambio al horario de verano.
- Se usó un índice numérico interno para statsmodels, conservando el índice temporal original para evaluación y visualización.
- Se incorporaron dummies de dia_semana para capturar diferencias entre días laborables, sábados y domingos.
- Se evaluaron dos variantes:
  a) SARIMAX base sin rezagos explícitos de demanda.
  b) SARIMAX + demanda_lag_168h como variante experimental enriquecida con dependencia semanal.

Resultados principales de diagnóstico:

- SARIMAX base: MAPE 11.16%
- SARIMAX + lag 168h: MAPE 2.56%
- Baseline lag 24h: MAPE 7.72%
- Baseline lag 168h: MAPE 1.60%

Interpretación:

El resultado mostró que SARIMAX base no capturaba adecuadamente la estructura semanal. La incorporación de demanda_lag_168h redujo drásticamente el error, confirmando la importancia de la dependencia semanal en la demanda eléctrica horaria.

2. sarimax_walk_forward.py
--------------------------
Función del archivo:

sarimax_walk_forward.py extendió el análisis de un único corte a varios cortes temporales distribuidos durante el año.

Cortes utilizados:

- 2025-02-05
- 2025-05-06
- 2025-07-08
- 2025-11-04

Modelos evaluados:

- SARIMAX base
- SARIMAX + lag 168h
- Baseline lag 24h
- Baseline lag 168h

Resultados medios iniciales:

- SARIMAX base: MAPE medio 6.47%
- SARIMAX + lag 168h: MAPE medio 3.69%
- Baseline lag 24h: MAPE medio 2.06%
- Baseline lag 168h: MAPE medio 5.32%

Interpretación:

SARIMAX + lag 168h mejoró claramente el rendimiento medio respecto a SARIMAX base. Sin embargo, el corte 2025-05-06 mostró un comportamiento problemático: el uso de la semana anterior como referencia empeoró la predicción.

3. sarimax_walk_forward_festivos.py
-----------------------------------
Función del archivo:

sarimax_walk_forward_festivos.py se creó para explicar metodológicamente el comportamiento anómalo del corte 2025-05-06.

La hipótesis era que la semana usada por demanda_lag_168h podía contener festivos que alteraban la comparación con el periodo de test.

El script añade al walk-forward:

- número de festivos en el periodo de test;
- número de festivos en la semana de referencia lag 168h;
- fechas festivas detectadas en test;
- fechas festivas detectadas en la semana lag 168h;
- diferencia de festivos entre test y semana lag.

Resultado relevante:

Para el corte 2025-05-06:

- Festivos test: sin festivos
- Festivos semana lag 168h: 2025-05-01

Esto explica que:

- Baseline lag 168h empeore hasta 11.15%.
- SARIMAX + lag 168h empeore hasta 7.93%.
- SARIMAX base funcione mejor en ese corte con 3.39%.

Interpretación metodológica
---------------------------
La variable demanda_lag_168h es una señal predictiva potente porque captura la demanda de la misma hora de la semana anterior. Sin embargo, es sensible a diferencias de calendario entre la semana de referencia y el periodo objetivo.

Cuando la semana anterior contiene festivos y el periodo test no, el lag semanal puede introducir una señal poco representativa.

Conclusión SARIMAX
------------------
Se conservan dos variantes conceptuales:

1. SARIMAX base
   - Modelo estadístico sin rezagos explícitos de demanda.
   - Útil como comparación estadística clásica.
   - Menos competitivo en la configuración evaluada.

2. SARIMAX + lag 168h
   - Variante enriquecida con dependencia semanal explícita.
   - Mejora el MAPE medio respecto a SARIMAX base.
   - Presenta mayor variabilidad y sensibilidad a semanas festivas o no comparables.

Estado metodológico
-------------------
Estado: evaluación walk-forward validada con diagnóstico de festivos.

Script recomendado para resultados:

sarimax_walk_forward_festivos.py

Archivo de resultados usado para la comparación:

src/models/SARIMAX/resultados_sarimax_walk_forward.csv
