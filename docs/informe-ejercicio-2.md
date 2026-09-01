# Ejercicio 2: historia experimental

**73.69 Large Language Models — Trabajo Práctico 1 — Ejercicio 2**

Cada número de este informe se puede localizar en `results/eda-contract/embeddings/selection.json`,
`results/eda-contract/audit.md` o un JSON de `results/eda-contract/`. Ningún número viene de memoria ni de
una corrida anterior con otras columnas.

## Índice

0. Tarea, target y lugar del Transformer
1. Partición agrupada y protocolo
2. Columnas que entrega el EDA
3. Técnicas de representación con el conjunto completo
4. Cotas del texto y escalera: L0a, L0, L1, L2, L0b
5. Complejidad y alternativas de módulos
5.1 Validación del recorrido
6. Curvas de entrenamiento y regularización
7. Selección de M y estabilidad por semillas
8. Transferencia como comparación secundaria
9. Evaluación final en holdout
10. Qué aprendimos y qué no demuestran los resultados

---

## 0. Tarea, target y lugar del Transformer

El target es `bought`: el modelo estima la probabilidad individual de compra y, por lo tanto, el BTR
esperado. Todos los candidatos reciben `title`, `description`, `ingredients`, `category`, `allergens` y
`price_position`, siempre juntos. Primero se elige una baseline clásica por comparaciones secuenciales de
representación (§3); después se compara esa referencia con embeddings aprendidos sin atención y con un
Transformer pequeño (§4); finalmente se prueban alternativas puntuales de sus módulos (§5), se confirma el
candidato con semillas y se abre el holdout una sola vez (§7 y §9).

## 1. Partición agrupada y protocolo

El split es por `query_id`: 20% holdout y 80% desarrollo; dentro de desarrollo, cinco folds agrupados y
estratificados (`src/model/experiment.partition`, sobre `src/partitions.build_query_partitions`). Varias
filas pertenecen a la misma búsqueda; separar por fila filtraría contexto entre train y validación, así que
se reserva por query. `tests/test_partitions.py::test_ninguna_query_cruza_particiones` verifica que ningún
`query_id` aparece en dos particiones a la vez, contra el dataset real. No se afirma que cada query
represente un único instante ni una única compra.

Con el dataset real: 10.000 filas, 2.012 queries, tasa positiva 13.01%, holdout de 2.002 filas (nunca
evaluado hasta §9), y folds de aproximadamente 6.400/1.600 filas cada uno.

## 2. Columnas que entrega el EDA

```text
Texto:               title + description + ingredients
Categoricas:          category + allergens
Numerica derivada:    price_position
Control diagnostico:  popularity_phrase, no elegible como entrada final
```

Congelado en `src/model/eda_contract.py` (`CONTRACT_FIELDS`, `DIAGNOSTIC_FIELDS`) y en
`parameters-eda.txt`. `require_valid` corre antes de cargar el dataset en todo entrypoint que acepta
`parameters-eda.txt`, y `tests/test_eda_contract.py` impide que un candidato pierda una columna o herede una
prohibida (`cart`, identificadores, timestamp, filtros, precio absoluto, `storage_type`,
`unit_of_measure`, `net_weight_oz`, `nutrition_score`).

## 3. Técnicas de representación con el conjunto completo

Ocho técnicas, tres familias, cinco folds cada una (40 filas en
`results/eda-contract/embeddings/linear-sweep.csv`), con un clasificador logístico fijo. En cada caso las
familias no evaluadas conservan su representación de referencia. El margen pareado (media ± un desvío de
las diferencias por fold) decide si una alternativa reemplaza la referencia; ver
`src/model/representation_selection.py`. No es un intervalo de confianza: los cinco folds comparten datos
de entrenamiento, así que la dispersión es descriptiva, no inferencia.

| Familia | Referencia | Alternativas probadas | Seleccionada | Motivo |
|---|---|---|---|---|
| Texto | bolsa binaria (AP 0.7710 ± 0.0191) | tf-idf (AP 0.7938 ± 0.0165) | **tf-idf** | mejora pareada clara |
| Categóricas | one-hot (AP 0.7710 ± 0.0191) | target encoding suavizado (AP 0.7547 ± 0.0207) | **one-hot** | el target encoding no mejora por el margen declarado |
| Numérica | buckets por cuantiles (AP 0.7710 ± 0.0191) | continuo estandarizado (AP 0.6899 ± 0.0205), continuo + buckets (AP 0.7716 ± 0.0190), piecewise-linear (AP 0.7790 ± 0.0200) | **piecewise-linear** | mejora pareada clara |

(`results/eda-contract/embeddings/selection.json` guarda esta misma decisión en formato máquina.)

Esta selección corre sobre el barrido lineal de Task 3 y **no** se traslada automáticamente a `L0`: `L0` es
la referencia lineal reproducible del contrato completo (bolsa de palabras + one-hot + buckets, igual que
`src/model/baseline.py`), y se reporta al lado el resultado del ganador exacto del barrido (tf-idf +
one-hot + piecewise) como evidencia de qué tan lejos se puede llevar la parte lineal si se lo permite.

## 4. Cotas del texto y escalera: L0a, L0, L1, L2, L0b

El EDA mostró que el título termina en una frase entre paréntesis (`popularity_phrase`) con tres niveles de
BTR muy separados (~65%, ~2%, exactamente 0%): es casi la clave del target. `L0a` mide el piso —qué queda
sin texto—, y `L0b` mide el techo del texto —qué se obtiene si alguien extrae la clave a mano—. Ninguna de
las dos es candidata ni entra a la búsqueda de arquitectura ni al holdout (`src/model/eda_contract.BRACKET_RUNS`,
verificado por `tests/test_eda_contract.py`).

| Rung | Descripción | AP media ± sd (5 folds) | ROC-AUC media ± sd |
|---|---|---:|---:|
| L0a | sin texto (piso) | 0.1812 ± 0.0133 | 0.6144 ± 0.0233 |
| L0 | texto crudo, lineal | 0.7710 ± 0.0191 | 0.9704 ± 0.0033 |
| L1 | embeddings aprendidos, sin atención | 0.7574 ± 0.0194 | 0.9666 ± 0.0039 |
| L2 | con autoatención | 0.7518 ± 0.0148 | 0.9647 ± 0.0053 |
| L0b | clave extraída a mano (techo) | 0.8130 ± 0.0083 | 0.9747 ± 0.0027 |

**L1 → L2 (¿aporta la atención?).** El margen pareado de `AP(L2) - AP(L1)` no es claramente positivo — la
media de L2 es menor que la de L1 y la comparación (`representation_selection.compare`) no arroja
`improves`. Por la regla de Task 4 Step 5: si el intervalo incluye cero, o si L2 pierde de forma clara, se
informa como resultado y **L1 queda como referencia de representación aprendida**, mientras que L2 sigue
siendo el Transformer requerido para las comparaciones modulares de §5 sin declararse superior. Este es un
resultado negativo que el informe reporta y no esconde dentro del peldaño siguiente: en esta reconstrucción
mínima (una sola capa, `d_model=64`, sin posición, pooling promedio) la autoatención no mostró todavía una
mejora medible sobre el promedio de embeddings.

**Fracción de recuperación.**

```text
recuperación = (AP(L2) - AP(L0a)) / (AP(L0b) - AP(L0a))
             = (0.7518 - 0.1812) / (0.8130 - 0.1812)
             ≈ 0.90
```

El denominador es claramente positivo (L0b supera a L0a por un margen enorme), así que la métrica se
publica: la escalera recupera aproximadamente el 90% de lo que separa "no leer nada" de "recibir la clave ya
extraída", leyendo únicamente el texto crudo. Esta lectura es auxiliar; PR-AUC sigue siendo la métrica
primaria.

## 5. Complejidad y alternativas de módulos

`src/model/run_architecture.py` recorre, desde L2: representación numérica → profundidad (1 vs 2 vs 3) →
ancho (32 vs 64 vs 96, desde la profundidad elegida) → heads (2 vs 4 vs 8, desde el ancho elegido) → posición
/ pooling / dropout. Cada etapa cambia exactamente un campo y se resuelve con el mismo margen pareado de §3;
profundidad y ancho usan `advance_complexity` (ratchet ordinal); heads usa `resolve_heads`, porque 8 no es
"más complejo" que 2 en ningún sentido útil — es una partición distinta de la misma atención, resuelta por
media más alta y, en empate, por la menor cantidad de heads. `tests/test_architecture_path.py` prueba que el
recorrido nunca genera las 27 combinaciones de `3 profundidades × 3 anchos × 3 heads`.

*(La corrida completa de `run_architecture.py` sobre el dataset real no se ejecutó en esta entrega — un
Transformer de cinco folds toma varios minutos cada uno y la búsqueda completa son hasta doce corridas; ver
§10, "Limitaciones". El código pasa sus propios tests con fixtures sintéticas exactas al plan.)*

### 5.1 Validación del recorrido

`src/model/run_greedy_validation.py` mide, después de elegir M y antes de abrir el holdout, dos sesgos del
descenso por coordenadas depth → width → heads: el sesgo de condicionamiento (ancho y heads nunca se
probaron con una profundidad distinta a la elegida) y el sesgo de orden (el punto final depende de haber
recorrido profundidad antes que ancho). La capa 1 reabre la profundidad descartada con los dos anchos y las
dos cantidades de heads (cuatro corridas nuevas); la capa 2 recorre los seis movimientos de un solo cambio
sobre el punto final en `n_layers`, `d_model` y `n_heads` — y cuesta cero entrenamientos si el recorrido
original nunca se movió de la base, porque entonces comparte digest con las seis corridas de §5. La
validación exige `improves`, nunca `tie-break`, para reemplazar a M: un desempate es aceptable durante la
búsqueda, pero M ya está congelado, y moverlo por una diferencia que el margen no resuelve sería cambiar de
finalista por ruido.

La validación evalúa hasta 17 de las 27 combinaciones de capacidad. Descarta dos sesgos concretos —el
condicionamiento de ancho y heads a una sola profundidad, y la dependencia del orden de los ejes— pero no es
una búsqueda exhaustiva y no demuestra que la configuración elegida sea el óptimo global de la grilla. El
margen pareado que decide cada comparación es una regla declarada, no un test.

## 6. Curvas de entrenamiento y regularización

Cada JSON de `results/eda-contract/` guarda, por fold, la curva de `train_ap`/`validation_ap` y la mejor
época (`curves`, ver `src/model/results.folds_with_diagnostics`). `run_eda_audit.py` resume esa evidencia por
corrida: mejor época, AP de entrenamiento en esa época y la brecha train−validation, para leer sobre- y
subajuste sin reentrenar nada.

## 7. Selección de M y estabilidad por semillas

`M` (`[M selected from directed comparisons]` en `parameters-eda.txt`) se declara copiando exactamente
`final_config` de `results/eda-contract/architecture/selection.json`; no se combina manualmente una
profundidad, un ancho y una cantidad de heads que no hayan aparecido juntos en el recorrido resuelto
(`src/model/run_eda_audit.py` lo audita). En esta entrega, mientras la búsqueda de arquitectura del dataset
real está pendiente (§10), `M` es una repetición explícita de `L2` — el comentario en `parameters-eda.txt` lo
dice y no se disfraza como una configuración distinta.

Dos repeticiones exactas miden la varianza de inicialización: `[S selected seed 7]` y
`[S selected seed 99]` copian `M` y cambian solamente `seed` (`changed_fields(M, S) == ("seed",)`,
verificado en `tests/test_eda_contract.py`). `src/model/run_declared.py --prefix "S selected"` las corre por
fuera de la escalera y del análisis de arquitectura.

Los finalistas están fijados de antemano en `src/model/eda_contract.FINALISTS`:

```python
FINALISTS = ("L0 linear raw EDA", "M selected from directed comparisons")
```

Ninguna cota diagnóstica (`L0a`, `L0b`) ni ninguna sonda de validación (`V `) puede ganar `run_final.select`
bajo el contrato, aunque score mejor en cross-validation (`tests/test_final.py`).

## 8. Transferencia como comparación secundaria

Las secciones `[T frozen MiniLM]` y `[T finetuned MiniLM]` de `parameters-eda.txt` declaran el conjunto
completo de columnas; el texto se serializa como `title + description + ingredients`, y `category`,
`allergens` y `price_position` se agregan con el mismo bloque tabular que usa la barra
(`tests/test_transfer.py::test_transfer_mantiene_el_conjunto_eda`). El encoder congelado (`MiniLM`) se
compara sobre cinco folds; el fine-tuning se reporta como un único fold y no se usa para afirmar
superioridad promedio sobre los modelos de cinco folds. *(La corrida real de `run_transfer.py` bajo el
contrato no se ejecutó en esta entrega — requiere descargar/cargar el checkpoint de `sentence-transformers`;
ver §10.)*

## 9. Evaluación final en holdout

*(Pendiente: requiere que la arquitectura de §5 esté resuelta con corridas reales y que `M` deje de ser una
repetición de `L2`. Una vez resuelto, `src.model.run_final --parameters parameters-eda.txt --results
results/eda-contract --final-results results/eda-contract/final` evalúa `L0` y `M` una sola vez cada uno.)*

## 10. Qué aprendimos y qué no demuestran los resultados

- La escalera L0a → L0 → L1 → L2 → L0b, corrida sobre el dataset real bajo el contrato de seis columnas,
  recupera ≈90% de lo que separa "no leer nada" de "recibir la clave ya extraída" leyendo únicamente texto
  crudo (§4). Ese es el resultado más sólido de esta entrega.
- En esta reconstrucción mínima la autoatención (L2) no superó de forma pareada al promedio de embeddings
  (L1): es un resultado negativo, reportado como tal, no oculto dentro de la arquitectura elegida.
- **Limitación explícita:** la búsqueda de arquitectura (§5), la validación de su recorrido (§5.1), la
  confirmación de `M` con semillas (§7) y la evaluación de holdout (§9) están implementadas y probadas con
  fixtures unitarias (`tests/test_architecture_path.py`, `tests/test_greedy_validation.py`,
  `tests/test_run_declared.py`, `tests/test_final.py`, todas en verde), pero **no se ejecutaron de punta a
  punta sobre el dataset real** en esta entrega: cada configuración neuronal de cinco folds toma varios
  minutos en CPU, y la búsqueda completa son hasta 15 configuraciones más 10 de validación más 2 semillas.
  Correr `scripts/run_all_experiments.py` (ver README) produce esa evidencia real; hasta entonces, `M` es
  formalmente una repetición de `L2` y el holdout no se ha abierto.
- La transferencia (§8) tiene el mismo estado: contrato verificado por test, ejecución real pendiente.
- El holdout no intervino en ninguna decisión de este informe.
