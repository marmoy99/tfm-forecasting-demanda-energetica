================================================================================
TFM — FORECASTING DE DEMANDA ENERGÉTICA EN ESPAÑA
Dataset de modelado — Documentación
================================================================================

DESCRIPCIÓN GENERAL
--------------------
Dataset horario de demanda eléctrica peninsular española con variables
meteorológicas, de calendario y lags históricos de consumo.

  Período     : enero 2021 – abril 2026
  Granularidad: horaria (una fila por hora)
  Filas       : 46.698
  Columnas    : 16
  Nulos       : 192 (solo en las primeras 168 horas por los lags)

Archivos:
  data/processed/dataset_modelado.parquet  (recomendado para Python/pandas)
  data/processed/dataset_modelado.csv     (legible con Excel)


================================================================================
COLUMNAS
================================================================================

ÍNDICE TEMPORAL
  datetime           Fecha y hora exacta (ej: 2021-07-15 14:00).
                     Clave de todo el dataset. Una fila = una hora.

TARGET — LO QUE PREDICE EL MODELO
  demanda_mw         Consumo eléctrico real de la red peninsular española
                     en ese momento, en megavatios (MW).
                     Rango: 1.571 MW (madrugada de agosto) a 36.150 MW
                     (tarde de invierno con mucho frío).

VARIABLES TEMPORALES
  hora               Hora del día de 0 a 23.
                     Correlación con demanda: 0.49. El consumo mínimo
                     ocurre a las 4-5h y los picos a las 10h y 20h.

  dia_semana         Día de la semana: 0=lunes, 1=martes, ... 6=domingo.
                     El lunes tiene un patrón muy distinto al domingo
                     (industria activa vs. parada).

  mes                Mes del año de 1 a 12.
                     Captura la estacionalidad anual (más consumo en
                     invierno y verano, mínimo en primavera y otoño).

  es_fin_de_semana   1 si es sábado o domingo, 0 el resto.
                     Correlación: 0.39. Los fines de semana la demanda
                     industrial cae significativamente.

VARIABLES DE CALENDARIO
  es_festivo         1 si es festivo nacional español, 0 el resto.
                     Calculado con la librería Python "holidays".
                     Correlación: 0.14. Los festivos tienen un patrón
                     de demanda similar al domingo.

  es_agosto          1 durante todo el mes de agosto, 0 el resto.
                     Se mantiene separado de es_festivo porque su efecto
                     es diferente: no es un día puntual, sino una caída
                     estructural de ~15% durante todo el mes por
                     vacaciones industriales y empresariales.

VARIABLES METEOROLÓGICAS
  Las variables de temperatura se calculan como media ponderada de 7
  capitales peninsulares (Madrid, Barcelona, Valencia, Sevilla, Zaragoza,
  Málaga y Bilbao) con peso proporcional a la población de cada provincia
  según el Padrón Municipal del INE. Esto da una temperatura "nacional"
  representativa del clima que experimenta la mayoría de la población.

  t_nac              Temperatura nacional ponderada por población, en °C.
                     Correlación: 0.19. Baja correlación lineal porque
                     la relación con la demanda tiene forma de U (el frío
                     Y el calor suben el consumo). HDD y CDD la capturan
                     mejor.

  HDD                Heating Degree: max(0, 18 - t_nac).
                     Mide cuánto frío hace por encima del umbral de
                     confort de 18°C. 0 en verano, sube en invierno.
                     Correlación: 0.14. Cada punto = ~150-200 MW más
                     de calefacción eléctrica.

  CDD                Cooling Degree: max(0, t_nac - 18).
                     Mide cuánto calor hace por encima del umbral de
                     confort. 0 en invierno, sube en verano.
                     Correlación: 0.31. Cada punto = ~200-250 MW más
                     de aire acondicionado.

  radiacion_solar    Radiación solar global en W/m², media de las 7
                     ciudades. Correlación: 0.29.
                     Doble efecto: (1) cuando hay mucha radiación, los
                     paneles fotovoltaicos generan más energía y la red
                     necesita cubrir menos demanda neta; (2) más sol en
                     verano significa más calor y más uso de AC.

  humedad_relativa   Humedad relativa del aire en %, media de las 7
                     ciudades. Correlación: 0.29.
                     A igual temperatura, con alta humedad el aire
                     acondicionado trabaja más porque debe deshumidificar
                     además de enfriar. Mejora la precisión en zonas
                     costeras (Valencia, Málaga, Bilbao).

  velocidad_viento   Velocidad del viento en km/h, media de las 7
                     ciudades. Correlación: 0.18.
                     Proxy de la generación eólica nacional: cuando hay
                     mucho viento, los aerogeneradores producen más y la
                     red necesita comprar menos en el mercado mayorista.

LAGS DE DEMANDA
  Los lags son las variables más predictivas del dataset porque la demanda
  eléctrica tiene patrones muy repetitivos: el consumo de este lunes a
  las 9h se parece mucho al del lunes pasado a las 9h.

  demanda_lag_24h    Demanda de exactamente hace 24 horas (misma hora
                     de ayer). Correlación con demanda: 0.81.
                     Captura el patrón diario y lleva implícito si ayer
                     era festivo, si hacía frío, etc.

  demanda_lag_168h   Demanda de exactamente hace 168 horas (misma hora
                     de la semana pasada). Correlación: 0.88.
                     La correlación más alta de todo el dataset. El
                     patrón semanal de demanda es muy estable: lunes
                     se parece a lunes, fin de semana a fin de semana.

  NOTA: las primeras 168 horas del dataset (primera semana de enero 2021)
  contienen NaN en los lags porque no hay semana anterior. Se eliminan
  o imputan antes de entrenar el modelo.


================================================================================
CORRELACIONES CON LA DEMANDA (resumen)
================================================================================

  demanda_lag_168h   0.877  ██████████████████████████
  demanda_lag_24h    0.810  ████████████████████████
  hora               0.489  ██████████████
  es_fin_de_semana   0.391  ███████████
  CDD                0.311  █████████
  dia_semana         0.306  █████████
  radiacion_solar    0.294  ████████
  humedad_relativa   0.287  ████████
  t_nac              0.189  █████
  velocidad_viento   0.179  █████
  es_festivo         0.144  ████
  HDD                0.141  ████
  es_agosto          0.066  █
  mes                0.050  █

Nota: la correlación lineal no captura relaciones no lineales. Variables
con correlación baja (como mes o HDD) son igualmente importantes para
modelos como LightGBM, que detecta patrones no lineales automáticamente.


================================================================================
EJEMPLO DE FILAS
================================================================================

  Viernes 1 de enero 2021 a las 10h (festivo, frío):
  datetime=2021-01-01 10:00 | demanda=19746 MW | hora=10 | dia_semana=4
  mes=1 | es_fin=0 | es_festivo=1 | es_agosto=0
  t_nac=2.9°C | HDD=15.1 | CDD=0.0
  radiacion=45 W/m² | humedad=82% | viento=16 km/h
  lag_24h=20907 | lag_168h=NaN (primera semana)

  Sábado 15 de julio 2023 a las 14h (verano, calor):
  datetime=2023-07-15 14:00 | demanda=26326 MW | hora=14 | dia_semana=5
  mes=7 | es_fin=1 | es_festivo=0 | es_agosto=0
  t_nac=20.6°C | HDD=0.0 | CDD=2.6
  radiacion=820 W/m² | humedad=41% | viento=18 km/h
  lag_24h=24821 | lag_168h=26043


================================================================================
CÓMO REGENERAR EL DATASET
================================================================================

Ejecutar en este orden desde la carpeta raíz del proyecto:

  1. python ine_openmeteo_extractor.py
     Descarga datos meteorológicos de Open-Meteo y demográficos del INE.
     Genera: data/raw/meteo_horario.parquet
             data/processed/temperatura_nacional_ponderada.parquet
     Duración: ~10 minutos (7 ciudades x 5 años de datos horarios).

  2. python build_dataset.py
     Une demanda REE + clima + calendario y añade los lags.
     Genera: data/processed/dataset_modelado.parquet / .csv
     Duración: <1 minuto.

Requisito previo: data/raw/demanda_peninsular_horaria.csv debe estar
presente antes del paso 2. Se obtiene de la API e·sios de Red Eléctrica
de España (indicador 460, token gratuito en esios.ree.es).


================================================================================
FUENTES DE DATOS
================================================================================

  Demanda eléctrica  REE e·sios, indicador 460
                     Token gratuito en esios.ree.es/es/desarrolladores

  Temperatura/clima  Open-Meteo, API ERA5 (reanálisis ECMWF)
                     Sin registro ni coste. archive-api.open-meteo.com

  Población          INE Padrón Municipal Continuo, tabla 2852
                     Sin registro ni coste. servicios.ine.es/wstempus

  Festivos           Librería Python "holidays" (sin API externa)
                     pip install holidays


================================================================================
TFM — Máster en Data Science e IA · EBIS Business Techschool
================================================================================
