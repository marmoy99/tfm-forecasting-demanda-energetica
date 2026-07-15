# Comparación SARIMAX clásico vs. SARIMAX armónico

## 1. Objetivo

El script `sarimax_comparacion_clasico_armonico.py` compara tres formulaciones
SARIMAX para la predicción horaria de la demanda eléctrica peninsular:

1. **SARIMAX clásico + lag 168h**
2. **SARIMAX armónico**
3. **SARIMAX armónico + lag 168h**

El objetivo del experimento es determinar si la representación explícita de las
estacionalidades mediante términos de Fourier mejora el rendimiento frente a una
formulación SARIMAX estacional clásica y si el lag semanal aporta información
adicional al modelo armónico.

La comparación se diseñó como un experimento controlado. Los tres modelos
comparten el mismo protocolo temporal, las mismas variables meteorológicas, el
mismo calendario, el mismo tratamiento de datos y las mismas métricas.

---

## 2. Motivación metodológica

La demanda eléctrica horaria presenta varias estacionalidades simultáneas:

- diaria: 24 horas;
- semanal: 168 horas;
- anual: aproximadamente 8.766 horas.

Un SARIMAX clásico dispone de una única estructura `seasonal_order`, por lo que
no puede representar directamente las tres periodicidades dentro de la parte
estacional del modelo.

La regresión armónica dinámica traslada las estacionalidades fuera del estado
SARIMAX y las representa como variables exógenas mediante pares de senos y
cosenos. De esta forma:

- Fourier representa los patrones periódicos;
- las variables meteorológicas y de calendario representan factores externos;
- el componente ARMA modela la autocorrelación residual;
- el lag de 168 horas incorpora el nivel observado en la misma hora de la semana
  anterior.

---

## 3. Modelos comparados

### 3.1. SARIMAX clásico + lag 168h

Configuración:

```python
order = (1, 1, 1)
seasonal_order = (1, 0, 0, 24)
trend = None
```

Variables:

- HDD;
- CDD;
- humedad relativa;
- velocidad del viento;
- radiación solar;
- festivo;
- fin de semana;
- agosto;
- dummies de día de semana;
- demanda observada 168 horas antes.

La estacionalidad diaria se modela dentro de SARIMAX y el patrón semanal se
refuerza mediante calendario y lag semanal.

### 3.2. SARIMAX armónico

Configuración:

```python
order = (2, 0, 2)
seasonal_order = (0, 0, 0, 0)
trend = "c"
```

Fourier:

```python
FOURIER = [
    (24, 6),
    (168, 3),
    (8766, 3),
]
```

Variables:

- términos Fourier diarios, semanales y anuales;
- meteorología;
- festivo;
- fin de semana;
- agosto;
- dummies de día de semana.

No utiliza demanda histórica como regresor explícito.

### 3.3. SARIMAX armónico + lag 168h

Utiliza la misma configuración del modelo armónico e incorpora:

```python
demanda_lag_168h
```

Esta variante combina:

- estructura estacional de largo plazo mediante Fourier;
- patrón reciente mediante la demanda de la semana anterior;
- meteorología;
- calendario;
- autocorrelación de corto plazo.

---

## 4. Protocolo común

Todos los modelos se evaluaron con:

- **12 cortes temporales**;
- **365 días de entrenamiento** por corte;
- **72 horas de horizonte**;
- separación de 30 días entre cortes;
- mismos regresores meteorológicos;
- mismo calendario nacional;
- mismo tratamiento de festivos;
- misma rejilla horaria;
- misma interpolación de huecos;
- misma estandarización de variables exógenas;
- métricas MAPE, MAE y RMSE;
- control de convergencia.

Los cortes abarcan aproximadamente un año completo y permiten representar
diferentes estaciones y patrones de consumo.

---

## 5. Preparación de datos

### 5.1. Rejilla horaria

El índice se regulariza con frecuencia horaria. Los huecos se interpolan
temporalmente para evitar discontinuidades incompatibles con el modelado.

### 5.2. Calendario

Se regeneran desde el índice:

- mes;
- hora;
- día de semana;
- fin de semana;
- agosto;
- festivo nacional.

### 5.3. Lag semanal

`demanda_lag_168h` se calcula después de regularizar la serie. Para un horizonte
de 72 horas, todos los valores del lag proceden del histórico observado.

### 5.4. Estandarización

Las exógenas se estandarizan con:

```python
media = exog_train.mean()
desviacion = exog_train.std()
```

Los estadísticos se calculan exclusivamente sobre train. El mismo escalado se
aplica al test, evitando fuga de información.

---

## 6. Métricas

### MAPE

Expresa el error porcentual absoluto medio.

### MAE

Expresa el error absoluto medio en MW.

### RMSE

Penaliza con mayor intensidad los errores grandes y también se expresa en MW.

El CSV generado mantiene un esquema común:

```text
corte, modelo, mape, mae, rmse
```

Esto facilita la construcción de tablas, gráficos, comparaciones estadísticas y
la integración con Prophet y LightGBM.

---

## 7. Resultados

| Modelo | MAPE medio | Desv. MAPE | Mediana MAPE | MAE medio | RMSE medio | MAPE máximo |
|---|---:|---:|---:|---:|---:|---:|
| SARIMAX armónico + lag 168h | **3,649 %** | **0,948** | 3,521 % | **918,813 MW** | **1.139,999 MW** | **5,619 %** |
| SARIMAX clásico + lag 168h | 3,957 % | 2,286 | **3,203 %** | 1.001,067 MW | 1.303,442 MW | 9,425 % |
| SARIMAX armónico | 4,221 % | 1,221 | 4,222 % | 1.064,061 MW | 1.335,420 MW | 6,522 % |

Ventanas ganadas por menor MAPE:

- SARIMAX clásico + lag 168h: 7 de 12;
- SARIMAX armónico + lag 168h: 3 de 12;
- SARIMAX armónico: 2 de 12.

Aunque el clásico gana más ventanas, presenta errores extremos considerablemente
mayores. El modelo armónico con lag ofrece el mejor equilibrio entre precisión
media y estabilidad.

---

## 8. Interpretación

### 8.1. Aporte de Fourier

El modelo armónico sin lag no supera al resto en promedio. Esto indica que
Fourier representa adecuadamente la estructura periódica, pero no sustituye por
completo la información reciente de la demanda.

### 8.2. Aporte del lag 168h

La incorporación del lag semanal mejora al modelo armónico en 9 de las 12
ventanas. El lag aporta información sobre el nivel reciente que no queda
totalmente recogida por calendario, meteorología y Fourier.

### 8.3. Estabilidad

El SARIMAX armónico + lag presenta:

- menor MAPE medio;
- menor MAE medio;
- menor RMSE medio;
- menor desviación estándar;
- menor error máximo.

La mejora media frente al clásico es de aproximadamente 0,31 puntos de MAPE,
equivalente a cerca de un 7,8 % relativo.

---

## 9. Anomalía del corte de abril de 2025

En el corte del 23 de abril de 2025, el modelo armónico sin lag obtiene un MAPE
de aproximadamente 1,98 %, mientras que las variantes con lag semanal empeoran:

- armónico + lag 168h: aproximadamente 5,62 %;
- clásico + lag 168h: aproximadamente 9,42 %.

La semana utilizada por `demanda_lag_168h` coincidió con Semana Santa. El patrón
de consumo de esa semana no era representativo de los días objetivo posteriores.
Por ello, el lag trasladó al horizonte una estructura de demanda atípica.

Este resultado no debe eliminarse del análisis. Constituye evidencia de una
limitación real de los lags semanales:

> Un lag de 168 horas es muy informativo en semanas comparables, pero puede
> degradar la predicción cuando la semana de referencia y el horizonte tienen
> calendarios laborales diferentes.

La presencia de Fourier reduce la dependencia exclusiva respecto del lag, lo que
explica que la variante armónica con lag falle menos que el modelo clásico en
ese corte.

---

## 10. Comparación estadística

Las pruebas pareadas de Wilcoxon entre las variantes no alcanzaron significación
estadística al nivel del 5 % con 12 ventanas.

Esto implica que no puede afirmarse una superioridad estadística concluyente.

La selección del candidato final se basa en criterios combinados:

- menor MAPE;
- menor MAE;
- menor RMSE;
- menor dispersión;
- menor sensibilidad a errores extremos;
- coherencia metodológica con las estacionalidades de la serie.

---

## 11. Decisión final

**Modelo seleccionado:** SARIMAX armónico + lag 168h.

**Estado:** validado como candidato final dentro de la comparación realizada.

**Justificación:** presenta el menor MAPE, MAE y RMSE medios, la menor dispersión
y una menor sensibilidad a errores extremos que el SARIMAX clásico.

**Limitación:** las diferencias entre modelos no son estadísticamente
significativas con las 12 ventanas disponibles.

---

## 12. Salida

El script genera:

```text
reports/model_comparison/resultados_sarimax_comparacion.csv
```

Columnas:

```text
corte
modelo
mape
mae
rmse
```

Número esperado de filas:

```text
12 cortes × 3 modelos = 36 filas
```

---

## 13. Dependencias

```bash
python -m pip install pandas numpy statsmodels holidays
```

---

## 14. Ejecución

Desde la raíz del repositorio:

```bash
python src/models/SARIMAX/sarimax_comparacion_clasico_armonico.py
```

---

## 15. Continuidad del pipeline

El modelo seleccionado se implementa posteriormente en:

```text
sarimax_armonico_final.py
```

El script final debe conservar exactamente:

- `order=(2,0,2)`;
- `trend="c"`;
- Fourier `(24,6)`, `(168,3)` y `(8766,3)`;
- lag 168h;
- regresores meteorológicos;
- calendario;
- estandarización con estadísticas del train;
- ventana final de 365 días.
