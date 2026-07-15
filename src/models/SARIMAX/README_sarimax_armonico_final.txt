# SARIMAX armónico final

## 1. Objetivo

El script `sarimax_armonico_final.py` entrena el modelo SARIMAX seleccionado como candidato final para la predicción horaria de la demanda eléctrica peninsular.

El modelo elegido es:

```text
SARIMAX armónico + lag 168h
```

La selección se realizó a partir de una comparación walk-forward entre tres variantes:

1. SARIMAX clásico + lag 168h.
2. SARIMAX armónico sin lag.
3. SARIMAX armónico + lag 168h.

El modelo armónico con lag obtuvo el mejor rendimiento agregado en MAPE, MAE y RMSE, además de una menor dispersión y una menor sensibilidad a errores extremos que el SARIMAX clásico.

Las diferencias entre modelos no resultaron estadísticamente significativas con 12 ventanas, por lo que la elección debe interpretarse como una selección basada en el conjunto de métricas y en la estabilidad, no como una superioridad estadística concluyente.

---

## 2. Configuración del modelo

La configuración final mantiene exactamente la lógica validada en el walk-forward comparativo:

```python
ORDER = (2, 0, 2)
SEASONAL_ORDER = (0, 0, 0, 0)
TREND = "c"
MAXITER = 300
```

La estacionalidad no se introduce mediante `seasonal_order`, sino mediante regresión armónica dinámica.

### Términos Fourier

```python
FOURIER = [
    (24, 6),
    (168, 3),
    (8766, 3),
]
```

Esto representa:

- estacionalidad diaria;
- estacionalidad semanal;
- estacionalidad anual.

Cada periodicidad se descompone en pares de seno y coseno.

---

## 3. Variables utilizadas

### Variables meteorológicas

- `HDD`
- `CDD`
- `humedad_relativa`
- `velocidad_viento`
- `radiacion_solar`

### Variables de calendario

- `es_festivo`
- `es_fin_de_semana`
- `es_agosto`
- dummies de `dia_semana`

### Variable autorregresiva externa

- `demanda_lag_168h`

Esta variable representa la demanda observada en la misma hora de la semana anterior.

Para un horizonte de 72 horas, todos los valores del lag proceden del histórico observado y no requieren predicciones recursivas.

---

## 4. Preparación de datos

El script reproduce la preparación utilizada durante el walk-forward comparativo.

### 4.1. Índice horario continuo

Se genera una rejilla horaria completa:

```python
pd.date_range(..., freq="h")
```

Los huecos temporales se interpolan para mantener una serie regular compatible con SARIMAX y con los términos Fourier.

### 4.2. Interpolación

Se interpolan temporalmente:

- `demanda_mw`
- HDD
- CDD
- humedad relativa
- velocidad del viento
- radiación solar

### 4.3. Regeneración del calendario

Las variables de calendario se vuelven a crear directamente desde el índice temporal para garantizar consistencia:

- mes;
- hora;
- día de semana;
- fin de semana;
- agosto;
- festivo nacional.

### 4.4. Cálculo del lag semanal

`demanda_lag_168h` se calcula después de regularizar la serie:

```python
df["demanda_lag_168h"] = df["demanda_mw"].shift(168)
```

---

## 5. Ventana de entrenamiento

El modelo final utiliza:

```text
365 días de entrenamiento
```

Esto equivale a:

```text
8.760 observaciones horarias
```

La ventana se toma desde el final del histórico disponible.

Esta decisión mantiene coherencia con el protocolo de validación y evita modificar la metodología después de seleccionar el modelo.

---

## 6. Construcción del horizonte futuro

El script genera un horizonte final de:

```text
72 horas
```

correspondiente a tres días completos.

Cuando la última observación del dataset no coincide con las 23:00 del día de corte, el script puede generar internamente las horas intermedias necesarias hasta alcanzar el inicio del horizonte objetivo.

Estas horas se pronostican. No se incorporan como observaciones reales ni se imputan como demanda conocida.

---

## 7. Meteorología futura

Como no se dispone de meteorología futura observada en el momento de generar la predicción, el script construye perfiles históricos usando únicamente la ventana de entrenamiento.

El procedimiento es:

1. media por mes y hora;
2. respaldo por hora;
3. respaldo global.

La metodología se identifica en el CSV mediante:

```text
perfil_historico_medio_mes_hora
```

Esta salida debe interpretarse como una predicción bajo meteorología estimada, no como una predicción operativa alimentada por un forecast meteorológico real.

---

## 8. Estandarización

Las variables exógenas se estandarizan utilizando exclusivamente estadísticas del train:

```python
media = exog_train.mean()
desviacion = exog_train.std()
```

El mismo escalado se aplica al horizonte futuro.

Esto evita fuga de información y mejora la estabilidad numérica del ajuste.

---

## 9. Control de convergencia

Después del entrenamiento se comprueba:

```python
resultado.mle_retvals.get("converged", False)
```

Si el modelo no converge, el script detiene la ejecución y no exporta la predicción final.

Esto evita guardar resultados numéricos de un ajuste no validado.

---

## 10. Archivos generados

### 10.1. CSV de métricas walk-forward

```text
reports/model_comparison/resultados_sarimax_armonico_final.csv
```

Columnas:

```text
corte
modelo
mape
mae
rmse
```

Este archivo contiene únicamente las 12 ventanas correspondientes al modelo seleccionado, filtradas desde el CSV comparativo.

Las métricas no pertenecen a la predicción futura. Representan el rendimiento validado del modelo en el walk-forward.

### 10.2. CSV de predicción final

```text
reports/predictions/prediccion_sarimax_armonico_final.csv
```

Orden de columnas:

```text
datetime
mes
dia_semana
hora
es_festivo
HDD
CDD
humedad_relativa
velocidad_viento
radiacion_solar
prediccion_demanda_mw
modelo
fecha_corte_entrenamiento
lags
meteo_futura_metodo
demanda_lag_168h
```

La estructura se mantiene alineada con el CSV final de LightGBM para facilitar:

- comparación entre modelos;
- concatenación de predicciones;
- creación de gráficos;
- integración con dashboards;
- trazabilidad de las variables futuras.

### Aclaración sobre `lags`

En este modelo:

```text
lags = 168
```

indica que se utiliza la demanda observada 168 horas antes como regresor exógeno.

No tiene exactamente el mismo significado que en LightGBM, donde `lags` representa la ventana autorregresiva completa utilizada por el forecaster.

### 10.3. Imagen de predicción

```text
src/models/SARIMAX/imagenes_generadas/sarimax_armonico_final.png
```

La figura muestra:

- últimos 7 días de demanda observada;
- predicción final de 72 horas;
- línea vertical en el corte de entrenamiento;
- eje temporal en intervalos de 12 horas.

---

## 11. Interpretación del modelo final

El modelo combina cuatro fuentes de información:

1. **Fourier**  
   Captura los patrones diarios, semanales y anuales.

2. **Lag de 168 horas**  
   Aporta información reciente sobre el nivel de demanda de la semana anterior.

3. **Meteorología**  
   Representa cambios en condiciones térmicas y ambientales.

4. **SARIMAX ARMA**  
   Modela la autocorrelación residual de corto plazo.

Esta combinación permite reducir la dependencia exclusiva del patrón de la semana anterior.

---

## 12. Limitación asociada al lag semanal

Durante la comparación walk-forward se observó una anomalía en abril de 2025.

La semana utilizada por `demanda_lag_168h` coincidió con Semana Santa, mientras que el horizonte posterior presentaba un patrón laboral distinto.

Esto provocó una degradación en las variantes con lag.

La conclusión metodológica es:

> El lag semanal es muy informativo cuando la semana de referencia y el horizonte son comparables, pero puede introducir sesgo cuando existen diferencias importantes de calendario.

El uso conjunto de Fourier y lag reduce parcialmente esta sensibilidad, pero no la elimina por completo.

---

## 13. Decisión metodológica registrada

**Modelo:** SARIMAX armónico + lag 168h.

**Estado:** validado como candidato final dentro de la comparación realizada.

**Justificación:** menor MAPE, MAE y RMSE medios, menor dispersión y menor sensibilidad a errores extremos que el SARIMAX clásico.

**Limitación:** las diferencias entre modelos no son estadísticamente significativas con las 12 ventanas disponibles.

---

## 14. Dependencias

```bash
python -m pip install pandas numpy matplotlib statsmodels holidays
```

---

## 15. Ejecución

Desde la raíz del repositorio:

```bash
python src/models/SARIMAX/sarimax_armonico_final.py
```

Al finalizar, la terminal informa:

- rango del train;
- horizonte objetivo;
- número de horas pronosticadas;
- rutas de los CSV;
- ruta de la imagen.

---

## 16. Validaciones mínimas

Antes de aceptar la salida final, comprobar:

- que el modelo converge;
- que el train contiene 8.760 filas;
- que la predicción exportada contiene 72 filas;
- que no hay nulos en las exógenas;
- que `fecha_corte_entrenamiento` coincide con el último dato observado;
- que `demanda_lag_168h` procede de timestamps históricos;
- que el horizonte exportado corresponde exactamente a tres días completos;
- que la imagen muestra continuidad temporal entre histórico y predicción.

---

## 17. Uso en la memoria

El script puede documentarse en las secciones:

- metodología;
- modelado estadístico;
- selección del modelo;
- validación temporal;
- predicción final;
- limitaciones;
- comparación con Prophet y LightGBM.

La redacción académica debe diferenciar claramente:

- métricas obtenidas en walk-forward;
- predicción futura sin demanda real disponible;
- meteorología estimada mediante perfiles históricos.
