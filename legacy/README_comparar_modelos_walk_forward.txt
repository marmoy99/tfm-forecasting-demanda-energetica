README - comparar_modelos_walk_forward.py

Objetivo del archivo
--------------------
El archivo comparar_modelos_walk_forward.py se creó para unir los resultados walk-forward de Prophet y SARIMAX en una tabla comparativa común.

La comparación no se realiza dentro de los scripts individuales de cada modelo para mantener separación de responsabilidades:

- Prophet genera sus propios resultados.
- SARIMAX genera sus propios resultados.
- comparar_modelos_walk_forward.py integra, valida y resume los resultados.

Entradas utilizadas
-------------------
El script lee los siguientes archivos CSV:

1. src/models/SARIMAX/resultados_sarimax_walk_forward.csv
2. src/models/Prophet/resultados_prophet_walk_forward_ventana365.csv

Estos archivos deben contener, como mínimo:

- corte
- modelo
- mape

Modelos comparados
------------------
La tabla comparativa incluye:

1. Prophet ventana 365
   Rol: modelo candidato.

2. SARIMAX base
   Rol: modelo estadístico sin rezagos explícitos de demanda.

3. SARIMAX + lag 168h
   Rol: variante enriquecida con dependencia semanal explícita.

4. Baseline lag 24h
   Rol: baseline diagnóstico.
   Nota: no debe interpretarse como modelo final para predicción multi-step directa, porque puede usar información real del propio periodo de test en horizontes superiores a 24 horas si no se implementa recursivamente.

5. Baseline lag 168h
   Rol: baseline semanal.
   Es una referencia importante porque usa la demanda observada de la misma hora de la semana anterior.

Salidas generadas
-----------------
El script genera dos archivos en la carpeta reports:

1. reports/model_comparison_walk_forward_by_cut.csv

Contiene la comparación de MAPE por corte temporal.

2. reports/model_comparison_walk_forward_summary.csv

Contiene el resumen por modelo:

- MAPE medio
- desviación estándar
- MAPE mínimo
- MAPE máximo
- número de cortes evaluados
- rol del modelo

Resultados principales
----------------------
Tabla resumen obtenida:

- Baseline lag 24h: MAPE medio 2.06%
- SARIMAX + lag 168h: MAPE medio 3.69%
- Prophet ventana 365: MAPE medio 4.05%
- Baseline lag 168h: MAPE medio 5.32%
- SARIMAX base: MAPE medio 6.47%

Interpretación
--------------
Prophet ventana 365 muestra un comportamiento estable entre cortes, con MAPE medio de 4.05% y baja variabilidad.

SARIMAX + lag 168h obtiene un MAPE medio menor, 3.69%, pero con mayor variabilidad. Su rendimiento depende de que la semana anterior sea comparable con el periodo de test.

SARIMAX base tiene peor rendimiento medio, 6.47%, lo que sugiere que la configuración estadística sin rezagos explícitos no captura suficientemente la dependencia semanal de la demanda horaria.

El baseline lag 24h obtiene el menor MAPE medio, pero se mantiene como baseline diagnóstico por el riesgo de comparación optimista en predicciones multi-step.

Razón metodológica de la comparación
------------------------------------
La comparación permite evaluar los modelos bajo condiciones comunes:

- mismos cortes temporales;
- mismo horizonte de predicción;
- misma ventana de entrenamiento de 365 días para Prophet y SARIMAX;
- misma métrica MAPE.

Esto aporta trazabilidad, reproducibilidad y una base más sólida para la discusión de resultados en la memoria del TFM.

Decisión sobre horizonte de predicción
--------------------------------------
Para esta fase se mantiene un horizonte de 3 días.

Justificación:

- El horizonte de 3 días permitió desarrollar, depurar y comparar los modelos de manera controlada.
- Es suficiente para analizar diferencias de comportamiento entre Prophet, SARIMAX base y SARIMAX + lag 168h.
- Evita introducir complejidad adicional antes de cerrar la comparación metodológica actual.

Estado metodológico
-------------------
Estado: comparación walk-forward validada para horizonte de 3 días.

Archivos finales de comparación:

- reports/model_comparison_walk_forward_by_cut.csv
- reports/model_comparison_walk_forward_summary.csv
