# LLM-TP1 — Buy Through Rate en un e-commerce de supermercado

Cómo correr el EDA, el Transformer y su evaluación.

## Entorno

### Linux / WSL

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

El `--extra-index-url` baja la wheel de PyTorch para CPU (~200 MB en vez de ~2 GB con
CUDA).

### Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

y reemplazar `.venv/bin/python` por `.venv\Scripts\python` en todo lo que sigue.

## EDA

```bash
.venv/bin/python -m src.eda.run_eda
```

Tarda unos minutos: las distribuciones nulas se estiman con 10.000 permutaciones por
columna (`src/eda/noise.py`).

## Modelo

Las corridas se declaran en `parameters-eda.txt`, no en el código: `[DEFAULT]` tiene la
arquitectura base y cada sección dice sólo lo que cambia.

> Hay dos archivos de parámetros y **no comparten ninguna sección**:
> `parameters-eda.txt` (53 secciones, el contrato del EDA) y `parameters.txt`
> (43 secciones, el barrido por ejes). El default es `parameters-eda.txt`, salvo en
> `run_transfer` y `run_modules`, que parten de secciones que sólo están en
> `parameters.txt`.

Cada configuración se mide con tres semillas (1337, 7, 99) y decide la media de las
tres.

### La escalera

```bash
.venv/bin/python -m scripts.run_ladder
.venv/bin/python -m scripts.run_seeds --prefix L1     # un peldaño, con las 3 semillas
```

### Representación de cada columna

```bash
.venv/bin/python -m scripts.run_embeddings --results results/v1-una-torre/eda-contract
```

Escribe `embeddings/linear-sweep.csv` y `embeddings/selection.json`.

### Búsqueda de arquitectura

```bash
.venv/bin/python -m scripts.run_bracket_search --results results/v1-una-torre/eda-contract
.venv/bin/python -m scripts.run_attention_ablation --results results/v1-una-torre/eda-contract
```

Escriben `architecture/bracket-search.json` y `architecture/attention-ablation.json`.

### Evaluación final

```bash
.venv/bin/python -m scripts.run_final_comparison --results results/v1-una-torre/eda-contract
.venv/bin/python -m scripts.run_ceiling_holdout --results results/v1-una-torre/eda-contract
```

**Se corren una vez.** Enfrentan dos modelos congelados antes de abrir el holdout, y
escriben `final/comparison.json` y `final/ceiling.json`.

### Transfer learning

```bash
.venv/bin/python -m scripts.run_transfer --parameters parameters.txt
```

### Figuras

```bash
.venv/bin/python -m scripts.run_eda_contract_figures \
    --results results/v1-una-torre/eda-contract --figures figures/eda-contract
.venv/bin/python -m scripts.run_figures \
    --results results/v1-una-torre/eda-contract --figures figures/final-bracket
```

Ninguna entrena: leen los JSON y las predicciones guardadas.

### Opciones comunes

```bash
--parameters otro.txt    # otro archivo de config (default: parameters-eda.txt)
--results otra/carpeta   # escribir en otro lado
--force                  # reentrenar aunque ya haya resultado guardado
```

## Resultados

Cada corrida deja un JSON en `results/<digest>.json` con la configuración resuelta, las
métricas por fold y las curvas por época. Una corrida ya registrada no se vuelve a
entrenar, así que un barrido se puede cortar y retomar.

```bash
.venv/bin/python -m src.model.results --directory results/v1-una-torre/eda-contract
```

```python
from src.model.results import summary_frame, fold_frame, curve_frame

summary_frame()   # una fila por corrida: media ± desvío de ROC y AP
fold_frame()      # una fila por fold
curve_frame()     # una fila por época
```

El `digest` no incluye el nombre de la sección, así que dos secciones con la misma
configuración comparten un solo registro. Una corrida se busca **por configuración y no
por nombre**.

Los pesos van a `results/weights/<digest>/fold-<k>.pt` y quedan fuera de git por tamaño.

## Tests

```bash
.venv/bin/python -m pytest tests -q
```

`tests/` está en `.gitignore`: es material de trabajo local y no se versiona.
