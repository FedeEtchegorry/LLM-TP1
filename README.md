# TP1 — Predicción de BTR en un e-commerce de supermercado

73.69 Large Language Models

## Ejercicio 1 — EDA

Análisis descriptivo del dataset organizado en siete aspectos. Toda conclusión
sale de agrupar filas y promediar `bought`: no hay modelos, correlaciones ni
tests de hipótesis.

```
py -3.13 -m src.eda.run_eda
```

Imprime las tablas de cada aspecto y escribe las figuras en `figures/`.

### Estructura

- `src/eda/loading.py` — lee el CSV y agrega las columnas derivadas.
- `src/eda/rates.py` — el único lugar donde se calcula una tasa de compra.
- `src/eda/plots.py` — los gráficos: por nivel, por tramo y el del ranking.
- `src/eda/noise.py` — cuánta separación produce el azar en cada columna.
- `src/eda/contribution.py` — cuánto separa una columna con otra fija al lado.
- `src/eda/report.py` — imprime tablas y medidas sueltas.
- `src/eda/aspects/` — un módulo por aspecto: sus tablas y sus figuras.
- `src/eda/ranking.py` — separación de cada columna contra su piso de ruido.
- `src/eda/run_eda.py` — corre los siete aspectos en orden y cierra con el ranking.

## Datos

`data/supermarket_products.csv`: 10.000 impresiones (un producto mostrado en una
búsqueda), 2.012 búsquedas, 22 columnas. El target es `bought`; el BTR global es
13,0%.
