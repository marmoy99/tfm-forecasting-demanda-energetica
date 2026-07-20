Prophet — predicción de demanda eléctrica

Tres scripts, cada uno con un único propósito.


1. prophet_evaluacion.py

Para qué sirve: ver un ejemplo concreto, con gráfico.

Entrena hasta una fecha (FECHA_CORTE) y predice los DIAS_TEST días siguientes,
que ya conocemos (son datos históricos). Compara la predicción con lo que
realmente pasó y calcula el MAPE de ese único corte.

Útil para: probar cambios rápido y ver una imagen de cómo predice el modelo.


2. prophet_walkforward.py

Para qué sirve: saber si el modelo es bueno de verdad. Este es el número oficial.

Hace lo mismo que prophet_evaluacion.py, pero repetido en varias fechas de
corte (CORTES), repartidas por distintas épocas del año. Al final da la
media de los MAPE de todos los cortes.

Por qué varios cortes y no uno: un único examen puede tocar una semana fácil
o difícil. Con varios, repartidos por el año, la nota es fiable de verdad.

No genera gráfico (solo números): mostrar 5 cortes × 72 horas cada uno
llenaría la pantalla sin aportar nada. Solo interesa el resumen.


3. prophet_final.py

Para qué sirve: la predicción real, la que se usaría en producción.

Entrena con todos los datos disponibles (sin dejar nada para comparar) y
predice los días siguientes al final del histórico, que son fechas que
todavía no han pasado.

Por eso este script no tiene MAPE: no existe el dato real con el que
comparar la predicción. Solo se puede mostrar el gráfico de lo que el modelo
cree que va a pasar.

Nota: el clima de esos días futuros no existe en nuestros datos. Se aproxima
repitiendo el último valor de clima conocido (en un sistema real, aquí iría
la previsión meteorológica).

Todos comparten el mismo dataset localizado en: data/processed/dataset_modelado.csv

Las imágenes se guardan en reports/figures.
Cada vez que se ejecuta un script que dibuja, sustituye la imagen anterior, porque se guarda siempre con el mismo nombre. 
Si quieres conservar varias, cambia el nombre del archivo en la variable correspondiente antes de ejecutar.