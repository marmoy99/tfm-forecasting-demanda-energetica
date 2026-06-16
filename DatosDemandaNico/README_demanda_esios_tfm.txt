================================================================================
TFM — DEMANDA ELÉCTRICA PENINSULAR DESDE E·SIOS
Notebook: 01_extraccion_demanda_esios_tfm.ipynb
Script auxiliar: fetch_esios_demanda.py
================================================================================

1. CONTEXTO
================================================================================

El objetivo del TFM es predecir la demanda eléctrica peninsular española hora a
hora. Para ello se necesita construir una variable objetivo sólida, reproducible
y alineada con el resto de datasets meteorológicos y demográficos.

Este bloque del pipeline descarga la demanda eléctrica desde e·sios / Red
Eléctrica, la limpia y la exporta como dataset horario.

La demanda resultante se usará como target del modelo:

    demanda_mw

A diferencia de la temperatura, la demanda peninsular no se pondera por
población. Ya representa una magnitud agregada del sistema eléctrico
peninsular.

================================================================================
2. FUENTE DE DATOS
================================================================================

Fuente:
    e·sios / Red Eléctrica

Endpoint:
    https://api.esios.ree.es/indicators/{indicator_id}

Indicador utilizado:
    1293 — Demanda real

Nivel geográfico:
    Península

Frecuencia:
    Horaria

Formato original:
    JSON vía API

Formatos exportados:
    CSV
    Parquet

Requisito:
    Token personal de e·sios definido como variable de entorno:

        export ESIOS_TOKEN="tu_token"

================================================================================
3. ESTRUCTURA DEL PIPELINE
================================================================================

Archivos principales:

    fetch_esios_demanda.py
        Script reutilizable de extracción, limpieza, validación y guardado.

    01_extraccion_demanda_esios_tfm.ipynb
        Notebook narrativo y ejecutable para documentar la extracción.

    build_dataset_modelado.py
        Script inicial para unir demanda con temperatura nacional ponderada.

Estructura de carpetas esperada:

    data/
    ├── raw/
    │   ├── demanda_esios_raw.csv
    │   └── demanda_esios_raw.parquet
    └── processed/
        ├── demanda_peninsular_horaria.csv
        ├── demanda_peninsular_horaria.parquet
        ├── temperatura_nacional_ponderada.csv
        ├── temperatura_nacional_ponderada.parquet
        ├── dataset_modelado_base.csv
        └── dataset_modelado_base.parquet

================================================================================
4. DATASET RAW
================================================================================

Nombre:
    demanda_esios_raw

Ruta:
    data/raw/demanda_esios_raw.csv
    data/raw/demanda_esios_raw.parquet

Descripción:
    Dataset normalizado directamente desde la respuesta JSON de e·sios.

Uso:
    Auditoría, trazabilidad y posible reprocesamiento.

No se recomienda usar este dataset directamente para modelado.

================================================================================
5. DATASET PROCESADO
================================================================================

Nombre:
    demanda_peninsular_horaria

Ruta:
    data/processed/demanda_peninsular_horaria.csv
    data/processed/demanda_peninsular_horaria.parquet

Descripción:
    Dataset horario limpio de demanda eléctrica peninsular.

Unidad:
    MW

Variable objetivo:
    demanda_mw

Clave temporal:
    datetime

Clave diaria para joins:
    fecha

Columnas principales:

    datetime
        Fecha y hora local normalizada sin timezone.

    datetime_utc
        Fecha y hora UTC original cuando está disponible.

    demanda_mw
        Demanda eléctrica en MW.

    geo_id
        Identificador geográfico de e·sios.

    geo_name
        Nombre geográfico. Debe corresponder a Península.

    tz_time
        Zona horaria reportada por e·sios.

    indicator_id
        Indicador de e·sios usado en la descarga.

    fecha
        Fecha en formato YYYY-MM-DD. Sirve para cruzar con variables diarias.

    año
        Año del registro.

    mes
        Mes del registro.

    dia
        Día del mes.

    hora
        Hora del día.

    dia_semana
        Día de la semana según pandas: lunes=0, domingo=6.

    es_fin_de_semana
        Indicador binario. 1 si sábado o domingo, 0 en otro caso.

    fuente
        Fuente del dato.

================================================================================
6. VALIDACIONES DE CALIDAD
================================================================================

Validaciones mínimas aplicadas:

    - datetime no nulo
    - demanda_mw no nula
    - sin duplicados por datetime
    - cálculo de horas faltantes
    - rango temporal mínimo y máximo
    - valores mínimo, máximo y medio de demanda

Ejemplo de reporte:

    rows
    start
    end
    duplicated_datetime
    missing_hours
    null_demanda_mw
    min_demanda_mw
    max_demanda_mw
    mean_demanda_mw

================================================================================
7. ENCAJE CON TEMPERATURA NACIONAL PONDERADA
================================================================================

La temperatura nacional ponderada viene del pipeline meteorológico y demográfico.
Esa variable sí está ponderada por población porque representa una agregación de
ciudades o zonas.

Relación entre datasets:

    demanda_peninsular_horaria
        frecuencia: horaria
        variable objetivo: demanda_mw

    temperatura_nacional_ponderada
        frecuencia: normalmente diaria
        variable predictora: temperatura ponderada por población

Primer join recomendado:

    demanda.merge(
        temperatura_nacional_ponderada,
        on="fecha",
        how="left",
        validate="many_to_one"
    )

Interpretación:

    muchas filas horarias de demanda se cruzan contra una fila diaria de
    temperatura nacional ponderada.

Si en el futuro se genera temperatura horaria nacional ponderada, el join
debería hacerse por datetime.

================================================================================
8. DATASET DE MODELADO BASE
================================================================================

Nombre:
    dataset_modelado_base

Ruta:
    data/processed/dataset_modelado_base.csv
    data/processed/dataset_modelado_base.parquet

Descripción:
    Tabla inicial para análisis exploratorio y primeros modelos.

Nivel de granularidad:
    Una fila por hora.

Variable objetivo:
    demanda_mw

Features temporales sugeridas:

    año
    mes
    dia
    hora
    dia_semana
    es_fin_de_semana
    hora_sin
    hora_cos
    mes_sin
    mes_cos

Features autoregresivas sugeridas:

    demanda_lag_24h
    demanda_lag_168h
    demanda_rolling_24h
    demanda_rolling_168h

Features meteorológicas:

    temperatura_nacional_ponderada
    HDD
    CDD
    variables meteorológicas agregadas disponibles

================================================================================
9. DECISIONES DE DISEÑO
================================================================================

1. Separar extracción, limpieza y modelado.
   El notebook documenta y ejecuta, pero la lógica reutilizable vive en scripts.

2. Guardar raw y processed.
   Raw permite auditar la fuente. Processed permite modelar.

3. Usar Parquet como formato principal.
   Parquet conserva tipos de datos y es más eficiente que CSV.

4. Mantener CSV como formato complementario.
   CSV facilita inspección manual y entrega académica.

5. No hardcodear token.
   El token se gestiona mediante variable de entorno.

6. No ponderar la demanda peninsular.
   La demanda ya es una serie agregada del sistema eléctrico peninsular.

================================================================================
10. REGLA PRÁCTICA PARA EL TFM
================================================================================

Demanda:
    target agregado del sistema eléctrico.

Temperatura:
    feature agregada y ponderada por población.

Tiempo:
    estructura principal del dataset.

Una fila:
    una hora.

Clave de modelado:
    datetime.

Clave auxiliar:
    fecha.

================================================================================
================================================================================
12. RESUMEN DE RETENCIÓN
================================================================================

Este notebook genera la variable objetivo del TFM:

    demanda_mw

La demanda se descarga desde e·sios, se filtra a nivel peninsular, se limpia a
frecuencia horaria y se guarda como dataset processed.

La temperatura nacional ponderada se une después como variable predictora.

La tabla final de modelado debe tener una fila por hora y combinar:

    demanda + calendario + clima + lags

================================================================================