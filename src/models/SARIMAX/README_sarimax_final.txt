README_sarimax_final.txt

Archivo asociado:
    src/models/SARIMAX/sarimax_final.py

Objetivo del script
-------------------
El script sarimax_final.py genera la predicción final del modelo SARIMAX para el escenario explícito acordado por el grupo:

    - Horizonte: 23, 24 y 25 de marzo de 2026.
    - Granularidad: horaria.
    - Corte límite de entrenamiento: 2026-03-22 23:00:00.
    - Ventana de entrenamiento: últimos 365 días disponibles antes del horizonte de predicción.

El objetivo no es realizar una evaluación contra datos observados, sino generar una predicción futura reproducible a partir de la información disponible hasta el corte definido.

Modelo utilizado
----------------
Se utiliza la variante:

    SARIMAX + demanda_lag_168h

Esta decisión se toma porque, en la evaluación walk-forward comparativa, SARIMAX + lag 168h fue la variante SARIMAX con menor MAPE medio frente al SARIMAX base:

    - SARIMAX base: MAPE medio 6.47%.
    - SARIMAX + lag 168h: MAPE medio 3.69%.

La variable demanda_lag_168h representa la demanda observada en la misma hora de la semana anterior. Su inclusión permite capturar dependencia semanal explícita, un patrón relevante en series horarias de demanda eléctrica.

Relación con Prophet final
--------------------------
La lógica general replica la filosofía del script prophet_final.py en estos puntos:

    - horizonte de predicción de 3 días;
    - predicción inmediatamente posterior al último dato disponible definido para el escenario;
    - entrenamiento con un histórico definido;
    - exportación de CSV y gráfica final.

La diferencia metodológica principal está en el tratamiento de las variables meteorológicas futuras. En lugar de repetir el último valor observado, sarimax_final.py estima las variables meteorológicas futuras mediante perfiles históricos medios.

Tratamiento de variables futuras
--------------------------------
Para construir el horizonte futuro se siguen estas reglas:

1. Calendario futuro
   Se genera directamente desde la fecha/hora futura:

       - hora;
       - día de semana;
       - mes;
       - es_fin_de_semana;
       - es_agosto;
       - es_festivo.

   La variable es_festivo se genera mediante la librería holidays para España. Si la librería no está instalada, el script solicita instalarla con:

       python -m pip install holidays

2. demanda_lag_168h
   Para cada hora futura se toma la demanda observada exactamente 168 horas antes. Por ejemplo:

       2026-03-23 00:00 usa la demanda observada de 2026-03-16 00:00.

   Este tratamiento es metodológicamente defendible para un horizonte de 3 días porque todos los valores de la semana anterior ya están observados al momento del corte de entrenamiento.

3. Variables meteorológicas futuras
   Las variables:

       - HDD;
       - CDD;
       - humedad_relativa;
       - velocidad_viento;
       - radiacion_solar;

   se estiman mediante un perfil histórico medio por mes y hora. Por ejemplo, para estimar una variable meteorológica futura en marzo a las 14:00, se utiliza la media histórica de esa variable para el mes 3 y la hora 14.

   Si no existiera una combinación mes-hora suficiente, el script utiliza como respaldo la media por hora y, en última instancia, la media global de la variable.

Justificación metodológica
--------------------------
La elección de SARIMAX + lag 168h mantiene coherencia con la evaluación walk-forward, donde esta variante fue la mejor opción SARIMAX en términos de MAPE medio. Además, conservar TRAIN_DIAS = 365 evita cambiar el régimen de entrenamiento respecto a la fase comparativa, controla el coste computacional e incluye un ciclo anual completo.

El horizonte de 3 días se mantiene porque fue el horizonte utilizado en la evaluación comparativa y porque permite construir demanda_lag_168h futura a partir de información observada de forma segura.

Limitaciones
------------
1. La meteorología futura no proviene de un forecast meteorológico real, sino de perfiles históricos medios. Esto mejora la lógica respecto a repetir el último valor observado, pero sigue siendo una aproximación.

2. La variable demanda_lag_168h puede verse afectada si la semana de referencia contiene festivos o patrones no comparables con el horizonte predicho. Por ese motivo el script imprime un diagnóstico de festivos para el horizonte futuro y su semana de referencia.

3. La predicción final no debe interpretarse como una evaluación de error, ya que no se comparan valores predichos contra observaciones reales dentro del propio script.

Salidas generadas
-----------------
El script guarda:

    reports/predictions/prediccion_sarimax_final.csv
    reports/figures/sarimax_final.png

El CSV incluye la predicción horaria y las variables futuras utilizadas por el modelo. Esto permite trazabilidad y revisión posterior.

Estado
------
Versión final operativa para el escenario de predicción a 3 días:

    2026-03-23 a 2026-03-25

Esta versión queda pendiente de mejora futura si se decide integrar predicciones meteorológicas reales, por ejemplo desde Open-Meteo.
