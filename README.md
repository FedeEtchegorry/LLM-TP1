# LLM-TP1 — Buy Through Rate en un e-commerce de supermercado

Predecir el **BTR** de un producto en una búsqueda: una fila del dataset es una
impresión, la salida es `P(bought)`, y el BTR de un producto es el promedio de esas
probabilidades sobre sus impresiones.

- **Ejercicio 1** — EDA que justifica target, features y preprocesamiento.
- **Ejercicio 2** — un Transformer propio (atención escrita a mano), con particiones,
  experimentos, métricas y estudio de ablación.
- **Ejercicio 3** — personalización, en la presentación.

---

## Preparar el entorno

### Linux / WSL

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

El `--extra-index-url` baja la wheel de PyTorch para CPU (~200 MB en vez de ~2 GB con
CUDA). Sin GPU no hace falta nada más.

Todos los comandos de abajo usan `.venv/bin/python` en vez de activar el entorno, para
que no dependan de que el `activate` haya corrido. Si preferís activarlo:

```bash
source .venv/bin/activate     # después alcanza con `python -m ...`
```

### Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

y reemplazar `.venv/bin/python` por `.venv\Scripts\python` en todo lo que sigue.

---

## Correr el EDA (Ejercicio 1)

```bash
.venv/bin/python -m src.eda.run_eda
```

Recorre los siete aspectos en orden (target, texto, precio, producto, envase,
composición, búsqueda), imprime las tablas de tasa de compra por nivel y por decil, y
al final el ranking de todas las columnas contra el piso de ruido.

- **Salida en consola**: las tablas, en español, listas para la presentación.
- **Figuras**: se escriben en `figures/` como PNG, numeradas por aspecto
  (`00-target-cart.png`, `01-texto-frase-titulo.png`, `02-precio-absoluto.png`, …).

Tarda unos minutos: el piso contra el que se compara cada columna se estima
remuestreando (`src/eda/noise.py`, 2000 barajadas por semilla), no se asume.

El análisis escrito está en [`docs/EDA.md`](docs/EDA.md), con sus figuras en
`docs/figures/`.

---

## Correr el modelo (Ejercicio 2)

**Todas las corridas se declaran en [`parameters.txt`](parameters.txt)**, no en el
código. `[DEFAULT]` tiene la arquitectura base y cada sección es una corrida que dice
sólo lo que cambia.

El archivo contiene **únicamente lo que un experimento mueve**: qué campos entran,
`d_model`, `n_layers`, `n_heads`, `dropout`, positional, pooling y el embedding
numérico. Épocas, batch, learning rate, weight decay, patience, semilla, cantidad de
buckets y la partición son constantes en `src/model/configs.py` — pero **entran igual al
digest** de cada corrida, así que cambiar una ahí invalida los resultados cacheados que
habría cambiado. Poner una de esas constantes a variar es moverla a `RunConfig` y al
archivo; nada más hay que tocar.

### La escalera

```bash
.venv/bin/python -m src.model.run_ladder
```

Los cinco peldaños, cada uno una corrida que es a la vez una diapositiva y un punto de
la tabla de ablación:

| | qué mide |
|---|---|
| `L0` | la barra: regresión logística, **AP 0.813 ± 0.008** |
| `L1` | texto sin atención (bag of embeddings) |
| `L2` | + atención — *¿aporta sobre un promedio?* (eje B) |
| `L3` | + columnas tabulares, números afines (eje G) |
| `L4` | + término de bucket en el embedding numérico (eje A) |

Hoy `L0` corre y da su número; `L1` a `L4` se reportan como *pending* hasta que llegue
`src/model/network.py`.

### El barrido de módulos *(pendiente)*

```bash
.venv/bin/python -m src.model.run_modules
```

Las 17 alternativas de módulo ya declaradas en `parameters.txt`, medidas contra `L4`:
profundidad, heads, `d_model`, positional encoding, pooling, campos de entrada y
dropout.

### Evaluación final *(pendiente)*

```bash
.venv/bin/python -m src.model.run_final
```

**Una sola** corrida sobre el 20% reservado, después de elegir la configuración por
validación cruzada. `evaluate_on_test` no se llama en ningún otro lado del repo.

### Opciones comunes

```bash
--parameters otro.txt    # correr con otro archivo de configuración
--results otra/carpeta   # escribir en otro lado
--force                  # reentrenar aunque ya haya resultado guardado
```

---

## Leer los resultados

Cada corrida deja un JSON en `results/<digest>.json` con la configuración resuelta, las
métricas por fold y las curvas por época. **Una corrida ya registrada no se vuelve a
entrenar**, así que un barrido se puede cortar y retomar.

```bash
.venv/bin/python -m src.model.results        # tabla de todo lo registrado
```

Para analizar desde Python:

```python
from src.model.results import summary_frame, fold_frame, curve_frame

summary_frame()   # una fila por corrida: media ± desvío de ROC y AP
fold_frame()      # una fila por fold: para las barras de error
curve_frame()     # una fila por época: over y underfitting
```

Los pesos entrenados van a `results/weights/<digest>/fold-<k>.pt` (se apagan con
`save_weights` en `Protocol`, la única constante que **no** entra al digest, porque
guardarlos o no es una decisión operativa y no debería invalidar nada). Quedan fuera de
git por tamaño; los JSON no, así que las tablas y figuras de la presentación se
regeneran sin reentrenar.

---

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

Lo que fijan: que ninguna `query_id` cruce un fold, que el test set nunca llegue a un
modelo, que vocabulario, niveles, medianas y bucket edges se ajusten **sólo** con las
filas de train, que `cart` no se lea nunca, y que nuestra atención coincida con
`nn.MultiheadAttention` cargándole los mismos pesos.

---

## Estructura

```
data/supermarket_products.csv   10.000 impresiones, 2.012 búsquedas
parameters.txt                  todas las corridas, declaradas
docs/EDA.md                     el análisis escrito (Ejercicio 1)
figures/                        figuras que genera run_eda
results/                        métricas por corrida (JSON) y pesos

src/partitions.py               StratifiedGroupKFold por query_id, 64/16/20
src/eda/                        el EDA: loading, rates, noise, contribution, aspects/
src/model/
    configs.py                  lee parameters.txt
    protocol.py                 folds, ROC-AUC y PR-AUC, evaluate_on_test
    baseline.py                 la regresión logística que fija la barra
    attention.py                atención escrita a mano, encoder-only
    encoding.py                 fila -> secuencia heterogénea
    results.py                  registro y cache de corridas
    run_ladder.py               los peldaños
```

Falta `network.py` (embeddings + encoder + cabezal `[CLS]`), `training.py` (loop por
fold con early stopping y curvas), y los entrypoints `run_modules.py` y `run_final.py`.

---

## Entrega

Repo + este README + hash de commit + presentación. La configuración exacta de cada
número reportado está en `parameters.txt` y en el `config` de cada
`results/<digest>.json`.
