# Supermarket BTR Forensics

**73.69 Large Language Models — Trabajo Práctico 1 — Ejercicio 1 (EDA)**

Before building anything, we took `data/supermarket_products.csv` apart. The dataset turns out to be
synthetic, and its generator is largely recoverable: **a hidden three-level popularity tier accounts for
most of the label**, one column leaks the answer outright, and the single most valuable transformation in
the whole problem is not about text at all.

| | |
|---|---|
| Product impressions | 10,000 |
| Search queries | 2,012 |
| `bought` — the positive class | 13.01% |
| Distinct tokens in `title` + `description` + `ingredients` | 422, none rare |
| Best linear baseline (average precision) | 0.813 ± 0.008 |

Every number in this document is produced by a script in `src/eda/`. See
[Reproducing](#reproducing) for the exact commands.

---

## 0. What is being predicted

Each row is one **product impression**: a product shown on a search-results page.

$$y_i = \begin{cases} 1 & \text{impression } i \text{ was bought} \\ 0 & \text{otherwise} \end{cases}
\qquad p_i = \hat{P}(y_i = 1 \mid \text{features available when the page renders})$$

Buy Through Rate is the aggregate of those impressions:

$$\text{observed BTR} = \frac{1}{N}\sum_i y_i = 0.1301
\qquad \text{predicted BTR} = \frac{1}{N}\sum_i p_i$$

So the model is a **row-level binary classifier**, and BTR for any group of impressions — a product, a
category, a price band — is the mean of $p_i$ over that group.

**How this identifies products to promote.** The assignment's goal is to find the best products and
promote them elsewhere in the e-commerce site. There is no product ID column, and 9,910 of 10,000 titles
are unique, so the model cannot memorise per-product history and a per-product BTR estimate would rest on
a single impression. Promotion therefore has to be driven by $p_i$ computed from **attributes** —
rank candidate products by their predicted purchase probability given their text, price and category.
This is a strength rather than a limitation: it generalises to newly listed products, which by
construction have no history at all.

The "features available when the page renders" clause is what excludes `cart` (§2).

---

## What we found

1. Every `title` and every `description` encodes the same hidden **popularity tier**, and they never
   disagree across all 10,000 rows. Tier A buys at 64.7%, tier B at 2.6%, tier C at exactly 0%.
2. `cart` is **target leakage** — no product was ever bought without being carted first. It must not be a
   feature.
3. The relationship between price and purchase is an **inverted U**, not a slope. Encoding `price_pct` in
   quantile buckets instead of as a linear term is worth **+0.137 average precision** — more than every
   other feature combined.
4. We found **no evidence of competition** between products in the same query: purchases per query are
   additive in the number of tier-A products.
5. A linear model over the popularity phrase, bucketed `price_pct`, category and allergens reaches
   **AP 0.813 ± 0.008**, matching an oracle given the hand-derived tier. That is the number the
   Transformer has to beat.

---

## 1. There is a hidden popularity tier, and it is written in the text twice

> **This accounts for most of the label.**

Every product title ends with a parenthetical — `(Customer Favorite)`, `(Standard Listing)` — and every
description ends with a sentence of the same flavour. Both encode one latent variable with three levels.
We checked all 10,000 rows for a title whose tier disagreed with its description's tier and found
**none**. The generator writes the tier twice and never contradicts itself.

| Tier | Phrases in the title | Rows | Bought | Rate |
|---|---|---:|---:|---:|
| **A** | Customer Favorite · Best Seller · Top Rated · #1 Pick | 1,931 | 1,249 | **0.647** |
| **B** | Well Reviewed · Shopper Favorite · Highly Rated · Popular Choice | 1,973 | 52 | 0.026 |
| **C** | the other 12 phrases · Limited Feedback · Clearance Listing · … | 6,096 | 0 | **0.000** |

**Tier C never buys — not once in 6,096 impressions.** That is a generator rule, not a statistical
tendency, and it is why the label is so learnable.

> **The A/B/C grouping is ours, not the data's.** We assigned it after looking at purchase rates over the
> whole dataset, so any model using it is an **oracle** and its score is an upper bound. The honest
> feature is the raw phrase, which is observable at impression time and can be learned from a training
> fold — see §8, where both are reported separately.

<details>
<summary>All 20 phrases, with their individual buy rates</summary>

| Phrase | Bought / rows | Rate | Tier |
|---|---:|---:|:--:|
| Customer Favorite | 334 / 493 | 0.678 | A |
| Best Seller | 309 / 470 | 0.657 | A |
| Top Rated | 296 / 472 | 0.627 | A |
| #1 Pick | 310 / 496 | 0.625 | A |
| Well Reviewed | 18 / 477 | 0.038 | B |
| Shopper Favorite | 14 / 507 | 0.028 | B |
| Highly Rated | 11 / 520 | 0.021 | B |
| Popular Choice | 9 / 469 | 0.019 | B |
| Limited Feedback | 0 / 524 | 0.000 | C |
| Standard Listing | 0 / 506 | 0.000 | C |
| *(no parenthetical)* | 0 / 511 | 0.000 | C |
| Rarely Reordered | 0 / 500 | 0.000 | C |
| Recently Added | 0 / 481 | 0.000 | C |
| Regular Listing | 0 / 494 | 0.000 | C |
| Clearance Listing | 0 / 510 | 0.000 | C |
| Low Feedback | 0 / 466 | 0.000 | C |
| Current Stock | 0 / 515 | 0.000 | C |
| Discontinuing Soon | 0 / 550 | 0.000 | C |
| Unrated Listing | 0 / 517 | 0.000 | C |
| New Listing | 0 / 522 | 0.000 | C |

The description's final sentence mirrors this exactly — *"Frequently reordered by returning customers"*
(0.668) and *"Rated highly by shoppers for consistent quality"* (0.660) for tier A, down to
*"Received mixed feedback from shoppers"* (0.000) for tier C.

</details>

![Bar chart of the buy rate of all 20 popularity phrases with 95% Wilson intervals: four bars near 0.65, four near 0.03, and twelve at exactly zero](figures/phrase-buy-rates.png)

*Buy rate per trailing parenthetical with a 95% Wilson interval. The gap between the four tier-A phrases
and everything else is far wider than the intervals, and the twelve zero-rate phrases are bounded above by
0.008 — with 6,096 impressions the sample rules out any real rate above that. This one figure is the
argument for keeping the phrase and the reason the label is learnable at all.*

### The phrases are deliberately confusable

Notice which words repeat across tiers:

```
"Customer Favorite"  → tier A  (64.7% buy)     "Top Rated"     → tier A
"Shopper Favorite"   → tier B  ( 2.6% buy)     "Highly Rated"  → tier B
          └── same word ──┘                       └─ same word ─┘
```

The shared word carries no information; the qualifier does. We expected this to defeat a bag-of-words
model, and it does not — `customer` and `shopper` are discriminative on their own. Adding bigram features
does not help either; it slightly hurts (§8). Word order is a weaker argument for attention here than it
first appears; the stronger argument is the numeric one in §4.

---

## 2. `cart` leaks the target and must be excluded

> **Drop this column.**

The cross-tabulation of `cart` against `bought` has an empty cell, and that empty cell is the whole story:

| `cart` | `bought` | rows |
|---|---|---:|
| false | false | 6,993 |
| false | true | **0** ← never happens |
| true | false | 1,706 |
| true | true | 1,301 |

![A 2x2 grid of cart against bought; the cart = false, bought = true cell is empty and marked "never happens"](figures/cart-leakage.png)

*The same table as a grid. The empty cell is the entire argument.*

Every purchase was carted first. `cart` is a later step of the same funnel — impression → add to cart →
buy — so a model given this column learns "not carted, therefore not bought" and scores beautifully on
paper. It would be useless in production: at the moment a search page renders, nothing has been carted
yet, so there is no value to supply. It fails the "available when the page renders" test in §0.

### What the leak is worth, which is the point

It is tempting to read `cart` as simply the best feature in the table. It is measured here so that the
temptation has a number attached to it:

| Feature set | ROC-AUC | PR-AUC (AP) |
|---|---:|---:|
| impression-time features (no `cart`) | 0.975 ± 0.003 | 0.813 ± 0.008 |
| the same features plus `cart` | 0.994 ± 0.001 | **0.946 ± 0.013** |
| `cart` alone | 0.904 ± 0.002 | 0.438 ± 0.004 |

![Bars of average precision: impression-time features at 0.813, the same plus cart at 0.946 and cart alone at 0.438, the last two in a separate colour as uncomputable at render time](figures/cart-leakage-ap.png)

*Adding `cart` buys +0.133 AP. Both red rows are uncomputable at the moment a search page renders, so the
gain is not available to any deployed model — it is the score of a feature that does not exist yet.*

**Decision.** `cart` is excluded from every feature set, and `src.eda.dataset.load_btr_data` does not
read it at all — the exclusion is structural rather than a convention to remember.
`tests/test_eda_dataset.py` hands the loader a file whose `cart` column cannot be parsed and asserts that
the load still succeeds: if the value were ever read, that test would fail.

---

## 3. The filter columns are copies of columns we already have

> **Drop three, derive one.**

The dataset was generated so that search results always satisfy the shopper's filter:

| Check | Rows satisfying |
|---|---:|
| `category == filter_category` | 10,000 / 10,000 |
| `storage_type == filter_storage_type` | 10,000 / 10,000 |
| `filter_price_min ≤ price ≤ filter_price_max` | 10,000 / 10,000 |

Serializing `filter_category` and `filter_storage_type` would lengthen every sequence to restate something
the model already knows. The price filter is different: it is always satisfied, but *where inside the
window* the price falls varies a great deal, and that turns out to matter enormously. So the three price
columns collapse into one derived feature:

$$\texttt{price\_pct} = \frac{\texttt{price} - \texttt{filter\_price\_min}}
{\texttt{filter\_price\_max} - \texttt{filter\_price\_min}}$$

Read it as: *the shopper asked for \$1.50–\$8.93 and this item costs \$4.29, so it sits 40% of the way up
their budget.*

![Buy rate per bin of price_pct over all rows, with confidence whiskers: a clear hump above the dataset average in the middle of the window and below it at both ends](figures/price-pct-derived.png)

*The three filter columns are satisfied by 10,000 rows out of 10,000, so as raw columns they carry nothing.
The quantity they hide — where inside the window the price falls — separates buyers from non-buyers even
before the popularity tier is held fixed. That is the trade: three columns out, one derived column in.*

---

## 4. Price follows an inverted U — the largest single win in the problem

> **This changes the architecture.**

Holding the tier fixed at A, purchase probability against `price_pct` rises from 0.35 to 0.87 and falls
back to 0.45. There is a plausible shopper story: an item at the very bottom of your price filter reads as
suspiciously cheap, one at the very top blows the budget, and the middle wins.

![Bars of P(bought) per decile of price_pct within tier A, rising from 0.35 to a peak of 0.87 near the middle of the window and falling back to 0.45](figures/price-inverted-u.png)

*P(bought) against `price_pct`, within tier A only (n = 1,931). Deciles of `price_pct`; shaded band is a
95% confidence interval. Descriptive plot over all tier-A rows.*

| Decile | Range | Bought / rows | Rate |
|---:|---|---:|---:|
| 0 | 0.02 – 0.13 | 68 / 193 | 0.352 |
| 1 | 0.13 – 0.21 | 79 / 193 | 0.409 |
| 2 | 0.21 – 0.29 | 123 / 194 | 0.634 |
| 3 | 0.29 – 0.37 | 137 / 192 | 0.714 |
| 4 | 0.37 – 0.44 | 163 / 193 | 0.845 |
| 5 | 0.44 – 0.52 | 167 / 193 | **0.865** |
| 6 | 0.52 – 0.60 | 167 / 194 | 0.861 |
| 7 | 0.60 – 0.68 | 141 / 192 | 0.734 |
| 8 | 0.68 – 0.77 | 116 / 193 | 0.601 |
| 9 | 0.77 – 0.95 | 88 / 194 | 0.454 |

### The other numeric columns are flat

The hump is not a property of numbers in general — it belongs to price. Every numeric column in the
dataset, including the three parsed purely so that dropping them is a measurement, is plotted on the same
axes:

![A grid of seven bar panels within tier A: price_pct and price both show a hump, while net_weight_oz, nutrition_score, package_value, volume_in3 and month_index are flat inside their confidence whiskers](figures/numeric-response-grid.png)

*P(bought) by decile of each numeric column, within tier A. `price_pct` and `price` carry the same hump —
one is derived from the other — and the remaining five stay inside their own noise around the tier average.
`package_value`, `volume_in3` and `month_index` are the parsed forms of `package_size`, `dimensions_in` and
`timestamp`, which §9 drops; this is the first of the two measurements behind that.*

### Why this dictates how numbers are embedded

A Transformer consumes tokens, and `8.30` is not a word. The standard fix is to give each numeric column
one sequence position whose vector is computed rather than looked up: `vector = value × w + b`. That
embedding is **affine in the value** — as the value rises, the vector traces a straight line through
embedding space, and the ordering of embeddings along that line is fixed.

This does *not* make the model's output monotonic: the encoder's feed-forward layers are non-linear and
can bend that line into a hump. The objection is about cost, not capability — recovering the shape spends
capacity that is scarce at 10,000 rows, when a lookup table represents it directly:

```python
emb = value * w_col + b_col        # exact precision, affine part
    + bucket_table[bin_of(value)]  # free-form, represents the U directly
```

§8 measures how much that second term is worth. It is the largest single effect in the problem.

---

## 5. No evidence of competition between products in a query

> **Negative result.**

A real recommendation system would model a search page as a choice among alternatives — a strong rival
should steal your purchase. We tested for it and found no sign of it. A tier-A product's buy rate is flat
no matter how many tier-A rivals share its query:

| Other tier-A products in the same query | Bought / rows | Rate |
|---:|---:|---:|
| 0 | 466 / 725 | 0.643 |
| 1 | 485 / 752 | 0.645 |
| 2 | 239 / 357 | 0.669 |
| 3 | 58 / 92 | 0.630 |

![Mean purchases per query rises linearly with the number of tier-A products](figures/purchases-additive.png)

*Purchases per query are additive: each tier-A product contributes its own ~0.65 purchases regardless of
what sits beside it. The chart shows counts 0–4; a single further query contains 5 tier-A products and is
omitted from the plot as one observation, but it is included in the table below.*

| Tier-A products in query | Queries | Total purchases | Mean |
|---:|---:|---:|---:|
| 0 | 768 | 17 | 0.02 |
| 1 | 725 | 492 | 0.68 |
| 2 | 376 | 493 | 1.31 |
| 3 | 119 | 240 | 2.02 |
| 4 | 23 | 58 | 2.52 |
| 5 | 1 | 1 | 1.00 |
| **total** | **2,012** | **1,301** | |

The maximum number of purchases *observed* in a single query is four; nothing in the data states a cap.

**What this does and does not license.** It is evidence against a competition effect, not proof that rows
are statistically independent — rows from one query still share a filter context and could correlate in
ways this test would not reveal. Two consequences follow:

- **Model each impression with a row-level binary output.** Cross-product attention over the candidates in
  a query would add real complexity for no measurable gain.
- **Keep query-grouped partitions anyway.** Rows from the same query share context, so splitting them
  across folds risks optimistic scores. The cost of grouping is nil.

---

## 6. `nutrition_score = 0` means "not applicable", not "terrible"

> **Data quality.**

1,244 rows carry a nutrition score of zero. They decompose exactly:

| Category | Rows with `nutrition_score = 0` |
|---|---:|
| Household | 642 |
| Personal Care | 602 |
| **total** | **1,244** |

![Histogram of nutrition_score with a tall isolated spike at zero, entirely coloured as Household and Personal Care, and a separate body of the distribution running from 18 to 99](figures/nutrition-sentinel.png)

*The spike at zero is disconnected from the rest of the distribution — nothing sits between 0 and 18 — and
every row inside it is Household or Personal Care. A sentinel, not a score.*

Shampoo does not have a nutrition score. The generator wrote `0` where it meant *not applicable*. But the
column's genuine range is **18–99**, so feeding a literal zero tells the model this is the worst-scoring
product in the catalogue — the opposite of the truth.

It is handled as missing: `load_btr_data` converts the sentinel to `NaN` and sets a `nutrition_missing`
flag; imputation then uses the **training fold's** median. Buy rate is unaffected by the sentinel (0.136
for zero-score rows vs 0.129 for the rest), so this is a correctness fix rather than a signal one.

---

## 7. PCA organises the data by food type, which is the question nobody asked

> **No preprocessing needed.**

We vectorised the text as a bag-of-words matrix and ran PCA on it. PCA maximises *variance*, not
predictive power, and the biggest source of variation in this corpus is what the product is — beverage,
snack, dairy.

Components are fitted on the training rows of fold 0 and each is scored on that fold's **validation** rows,
so the ranking below is a measurement rather than a description of the rows it was fitted on.

| PC | Variance | AUC vs `bought` | Heaviest words |
|---:|---:|---:|---|
| 1 | 5.7% | 0.510 | natural, flavors, prepared, ingredients, oz |
| 2 | 4.9% | 0.564 | the, of, aisle, items, its |
| 3 | 4.8% | 0.541 | ambient, of, water, salt, the |
| 4 | 4.1% | 0.525 | refrigerated, filtered, beverages, fl, salt |
| **5** | 3.8% | **0.778** | by, listing, feedback, limited, customers |
| 6 | 3.5% | 0.576 | salt, lb, sugar, water, bakery |
| **7** | 3.3% | **0.679** | regular, part, as, feedback, aisle |
| 8 | 2.8% | 0.521 | oil, snacks, grains, seasoning, ambient |
| 9 | 2.6% | 0.561 | to, catalog, feedback, the, pick |
| **10** | 2.2% | **0.606** | customers, reordered, shoppers, rated, highly |

![Two scatter plots of the first two principal components: coloured by tier, all three tiers are mixed inside every cluster; coloured by bought, purchases are spread across all clusters](figures/pca-text.png)

*The clusters are product categories, and each one contains all three tiers. PC1 and PC2 together explain
only 10.6% of variance and separate beverages from snacks — not buyers from non-buyers. Projection fitted
over all rows for display; the ranking table above uses train-fitted components scored on held-out rows.*

The largest components are worth nothing (AUC ≈ 0.51–0.56). The predictive signal lives in components 5, 7
and 10, and even the best of them reaches only **0.778**.

**Do not use PCA as a preprocessing step:** handing the model PC1–PC12 would feed it the food-category
axes and dilute what matters.

Two smaller results from the same analysis:

- **12 of the 422 tokens appear in every single row** — `a, and, for, grocery, in, intended, listed,
  online, orders, package, storage, under`. Zero variance, zero information; they are template
  scaffolding.
- **PCA on the numeric columns finds nothing** (best component AUC 0.549). The only strong pair is
  `price` ↔ `price_pct` at 0.711, which is expected since one derives from the other; the next two are
  `net_weight_oz` ↔ `volume_in3` at 0.38 and `nutrition_score` ↔ `package_value` at −0.23, both mild and
  both unsurprising — a bigger box holds more ounces. Everything else is under 0.15. The numeric effects
  are conditional on tier, and a linear unsupervised method is blind to that by construction.

![Correlation matrix of the seven numeric columns: the diagonal, one strong off-diagonal pair between price and price_pct at 0.71, a mild 0.38 between net_weight_oz and volume_in3, and everything else near zero](figures/numeric-correlation.png)

*Pearson correlation over the seven numeric columns. Only one pair is strongly related and it is the
derived one, so there is nothing here for a rotation to compress: PCA over these columns is a change of
basis with no compression to offer.*

---

## 8. What a Transformer actually has to beat

> **The bar to clear.**

### Protocol

Stated once, so every number below is unambiguous.

| | |
|---|---|
| Partitioning | `src.partitions.build_query_partitions`, whole `query_id` groups |
| Test set | 20% (2,002 rows), one outer `StratifiedGroupKFold(5)` fold, **never scored here** |
| Cross-validation | inner `StratifiedGroupKFold(5)` over the remaining 8,000 rows |
| Per fold | train 6,401 rows (64%) · validation 1,597 rows (16%) |
| Seeds | 42 for the outer split, 43 for the inner |
| Reported | mean ± standard deviation over the 5 validation folds |
| Model | `sklearn.linear_model.LogisticRegression`, L2, `C = 1.0`, lbfgs |
| ROC-AUC | `sklearn.metrics.roc_auc_score` |
| PR-AUC | `sklearn.metrics.average_precision_score` — the step-wise sum $\sum_n (R_n - R_{n-1})P_n$. Called "average precision"; **not** a trapezoidal area under an interpolated curve |
| Fitted per fold on training rows only | vocabulary, categorical levels, imputation medians, scaler mean and scale, quantile bucket edges |

Unseen categorical levels transform to an all-zero row and unseen words are dropped, so a validation row
can never introduce a new column. `tests/test_eda_features.py` asserts each of these properties.

PR-AUC is the metric to watch; with 13% positives, ROC-AUC flatters everything.

### Baselines — impression-time features only

| Model | ROC-AUC | PR-AUC (AP) |
|---|---:|---:|
| random | 0.500 | 0.130 |
| numerics only, linear | 0.546 ± 0.030 | 0.149 ± 0.014 |
| tabular only (numerics + categoricals) | 0.567 ± 0.023 | 0.158 ± 0.005 |
| bag-of-words, unigrams | 0.960 ± 0.003 | 0.671 ± 0.017 |
| bag-of-words + bigrams | 0.956 ± 0.005 | 0.646 ± 0.030 |
| bag-of-words + tabular | 0.964 ± 0.003 | 0.693 ± 0.014 |
| bag-of-words + tabular + **bucketed numerics** | 0.970 ± 0.003 | 0.759 ± 0.014 |
| popularity phrase only | 0.956 ± 0.002 | 0.649 ± 0.006 |
| **popularity phrase + bucketed `price_pct` + category + allergens** | **0.975 ± 0.003** | **0.813 ± 0.008** |

Three things to read off this table:

1. **Reading the tier out of raw text is easy.** Bag-of-words alone reaches ROC 0.960 — a linear model on
   422 word-columns does it without help.
2. **Bigrams do not help.** They cost 0.025 AP. The "confusable phrases" trap is handled by unigrams.
3. **Bucketing the numerics is worth +0.066 AP** on the bag-of-words model (0.693 → 0.759), and the
   focused feature set reaches 0.813.

### Oracle headroom — upper bounds, not achievable results

These rows use the hand-assigned A/B/C tier from §1, whose grouping was chosen by inspecting whole-dataset
purchase rates. They exist to locate the headroom, and **must not be quoted as model performance.**

| Model *(oracle)* | ROC-AUC | PR-AUC (AP) |
|---|---:|---:|
| *oracle tier only* | 0.954 ± 0.004 | 0.634 ± 0.023 |
| *oracle tier + `price_pct`, linear* | 0.959 ± 0.004 | 0.633 ± 0.030 |
| *oracle tier + `price_pct` in 10 buckets* | 0.971 ± 0.004 | **0.770 ± 0.023** |
| *+ category* | 0.973 ± 0.002 | 0.798 ± 0.013 |
| *+ allergens* | 0.975 ± 0.003 | 0.810 ± 0.009 |
| *+ brand + country* | 0.975 ± 0.003 | 0.806 ± 0.010 |

**The bucket encoding is worth +0.137 AP** (0.633 → 0.770) over a linear term on the *same* feature. That
is the single largest effect in the problem, and it is a pure encoding choice — no new information.

**Brand and country add nothing**; including them costs 0.004 AP. Drop them.

**The oracle tier is not needed.** The best honest baseline (0.813 ± 0.008) matches the best oracle
(0.810 ± 0.009). The 20-value popularity phrase, learned from training labels, carries everything the
hand-derived grouping did — so the A/B/C mapping is an analysis aid, not a feature.

### The framing for the presentation

A linear model over impression-time features reaches **AP 0.813 ± 0.008 / ROC 0.975 ± 0.003**. That is the
bar.

The design hypothesis for Ejercicio 2, stated as a hypothesis because the Transformer has not been
evaluated yet:

> A Transformer that reads the product text as tokens and embeds numeric columns with a bucket component
> alongside the affine term should **match or exceed the 0.813 linear baseline**. If it lands near 0.69 —
> the bag-of-words + linear-numerics figure — the numeric embedding is the component at fault, not the
> text encoder.

The ablation that tests it is already implied by the table: **linear numeric embedding vs linear + bucket**,
holding everything else fixed. `src.eda.evaluation.evaluate_across_folds` takes any
`score_fold(train_indices, validation_indices) -> scores` callable, so the Transformer reports into the
same table, on the same folds, under the same protocol.

---

## 9. Feature disposition

> **Carried into the model.**

Every row of the table at the end of this section is the conclusion of a measurement, not a judgement. Two
panels produce most of them, both under the §8 protocol — five query-grouped folds, logistic regression,
mean ± standard deviation — and both include the three columns the analysis drops, parsed for exactly this
purpose: `package_value` (from `package_size`), `volume_in3` (from `dimensions_in`) and `month_index`
(from `timestamp`). A column cannot be dismissed for carrying nothing until it has been given the chance
to carry something.

### What each column is worth alone

![Bars of average precision per column with error bars: text and popularity_phrase far above the rest near 0.67 and 0.65, price_pct at 0.167, and a long tail of columns sitting on the 0.130 random line](figures/variable-univariate-ap.png)

| Column, alone | ROC-AUC | PR-AUC (AP) |
|---|---:|---:|
| `title` + `description` + `ingredients` *(keep)* | 0.960 ± 0.003 | 0.671 ± 0.017 |
| `popularity_phrase` *(keep)* | 0.956 ± 0.002 | 0.649 ± 0.006 |
| `price_pct` *(keep)* | 0.584 ± 0.013 | 0.167 ± 0.009 |
| `category` *(keep)* | 0.541 ± 0.017 | 0.148 ± 0.007 |
| `allergens` *(keep)* | 0.553 ± 0.023 | 0.147 ± 0.009 |
| `price` *(keep)* | 0.545 ± 0.022 | 0.142 ± 0.011 |
| `net_weight_oz` *(keep)* | 0.509 ± 0.031 | 0.139 ± 0.016 |
| `unit_of_measure` *(keep)* | 0.518 ± 0.024 | 0.137 ± 0.007 |
| `volume_in3` *(drop)* | 0.506 ± 0.024 | 0.133 ± 0.009 |
| `country_of_origin` *(drop)* | 0.502 ± 0.010 | 0.133 ± 0.002 |
| `month_index` *(drop)* | 0.490 ± 0.018 | 0.132 ± 0.006 |
| `brand` *(drop)* | 0.498 ± 0.027 | 0.132 ± 0.010 |
| `nutrition_score` *(keep)* | 0.504 ± 0.021 | 0.130 ± 0.004 |
| `storage_type` *(keep)* | 0.500 ± 0.019 | 0.130 ± 0.005 |
| `package_value` *(drop)* | 0.478 ± 0.016 | 0.125 ± 0.005 |

Random is AP 0.130 and ROC 0.500. Read down the column: **the five dropped columns all land on the random
line**, three of them below it. `brand` reaches ROC 0.498 — a coin flip over 25 levels.

Two kept columns also sit on the random line, `nutrition_score` and `storage_type`. They are kept anyway,
for reasons the next panel does not measure: `nutrition_score` is one token that carries a documented
data-quality fix (§6), and `storage_type` is one token that costs a sequence position. Both are cheap, and
both are ablatable in Ejercicio 2 — which is the honest way to keep a column that a single measurement
cannot justify.

### What each column adds that the others do not

Alone is not the same as useful. A column can be worthless by itself and still contribute in company, or
be strong alone and entirely redundant next to a stronger one. The complement of the first panel removes
one column at a time from the full tabular set:

![Bars of the change in average precision when each column is removed: popularity_phrase at -0.620, price_pct at -0.044, allergens at -0.022, and every remaining column within +/- 0.006 of zero](figures/variable-leave-one-out.png)

| Feature set | ROC-AUC | PR-AUC (AP) | Δ AP |
|---|---:|---:|---:|
| full tabular set | 0.971 ± 0.003 | 0.783 ± 0.013 | +0.000 |
| without `popularity_phrase` | 0.584 ± 0.020 | 0.163 ± 0.012 | **−0.620** |
| without `price_pct` | 0.968 ± 0.003 | 0.739 ± 0.026 | −0.044 |
| without `allergens` | 0.969 ± 0.002 | 0.761 ± 0.015 | −0.022 |
| without `volume_in3` | 0.972 ± 0.002 | 0.781 ± 0.011 | −0.002 |
| without `unit_of_measure` | 0.971 ± 0.003 | 0.782 ± 0.013 | −0.001 |
| without `country_of_origin` | 0.971 ± 0.002 | 0.783 ± 0.014 | −0.000 |
| without `storage_type` | 0.971 ± 0.003 | 0.783 ± 0.015 | +0.000 |
| without `net_weight_oz` | 0.972 ± 0.003 | 0.784 ± 0.018 | +0.001 |
| without `package_value` | 0.972 ± 0.003 | 0.784 ± 0.015 | +0.001 |
| without `nutrition_score` | 0.971 ± 0.003 | 0.785 ± 0.013 | +0.002 |
| without `brand` | 0.972 ± 0.002 | 0.785 ± 0.011 | +0.002 |
| without `category` | 0.971 ± 0.003 | 0.785 ± 0.021 | +0.002 |
| without `price` | 0.972 ± 0.003 | 0.787 ± 0.009 | +0.004 |
| without `month_index` | 0.972 ± 0.003 | 0.789 ± 0.016 | **+0.006** |

Three columns cost something to remove; everything else is free. Two entries deserve a note, because they
disagree with the first panel:

- **`category` is worth +0.028 AP in §8 but ≈ 0 here.** The §8 row adds `category` to a *narrow* feature
  set — phrase plus bucketed `price_pct`. In the full set its contribution is already carried by
  `allergens` and `unit_of_measure`, which correlate with what a product is. The column is kept because it
  helps where it is used, and the honest reading is that it is redundant, not uninformative.
- **`price` is redundant with `price_pct`, not useless.** It is strong alone (AP 0.142, and the hump in
  §4 is visible in both) and free to remove, because `price_pct` is a function of it. Both are kept —
  one raw, one derived — and the pair costs two sequence positions.

Every removal that lands at or above zero is a column whose deletion the data does not object to. All five
dropped columns are in that group, `month_index` at the far end: removing `timestamp` *improves* the mean
AP by 0.006, within one standard deviation of zero.

### Categorical levels, one interval at a time

The leave-one-out panel says `brand` and `country_of_origin` add nothing. The reason is visible without any
model:

![Buy rate per level of brand and country_of_origin with 95% Wilson intervals; 24 of the 25 intervals cross the 13% dataset average](figures/levels-dropped.png)

*25 levels across the two columns (15 brands, 10 countries) and **24 of their 25 intervals contain the
0.1301 dataset average**. The exception is `North Star Foods` at 0.158 (664 rows, interval 0.132–0.188),
which clears the average by 0.002 — one marginal escape out of 25 independent 95% intervals is what chance
alone produces, and the leave-one-out panel confirms it: removing `brand` costs nothing. There is no level
here worth a coefficient.*

![Buy rate per level of category, allergens, storage_type and unit_of_measure with 95% Wilson intervals; several category and allergen levels sit clear of the dataset average](figures/levels-kept.png)

*The kept categoricals, on the same axes. Eight levels clear the average by more than their interval —
`Baby` (0.191), `Bakery` (0.164) and `Seafood` (0.057) in `category`, and five allergen levels running from
`Wheat` (0.155) down to `Shellfish` (0.055). Small effects next to the popularity phrase, but real ones,
and they are what the +0.028 and +0.012 AP rows in §8 are made of. `storage_type` and `unit_of_measure`
separate nothing, which matches their flat rows in both panels above; they are kept as one cheap token
each, not on this evidence.*

| Column | Disposition | Reason |
|---|---|---|
| `title` | keep — tokenised | carries the popularity phrase |
| `description` | keep — tokenised | carries the phrase a second time; redundancy is ablatable |
| `ingredients` | keep — tokenised | short, cheap, plausibly informative |
| `price` | keep — affine + bucket | inverted U |
| `price_pct` | **derive** — affine + bucket | strongest numeric feature; bucketing worth +0.137 AP |
| `net_weight_oz` | keep — affine + bucket | weak but independent of everything else |
| `nutrition_score` | keep + missing flag | 0 is a sentinel for Household and Personal Care |
| `category` | keep — categorical token | +0.028 AP over tier and price |
| `allergens` | keep — categorical token | +0.012 AP |
| `storage_type` | keep — categorical token | cheap, one token |
| `unit_of_measure` | keep — categorical token | cheap, one token |
| `bought` | **target** | the label |
| `cart` | **drop** | target leakage — bought ⟹ carted, always |
| `filter_category` | **drop** | identical to `category` in 10,000/10,000 rows |
| `filter_storage_type` | **drop** | identical to `storage_type` in 10,000/10,000 rows |
| `filter_price_min` / `filter_price_max` | **drop after deriving** | folded into `price_pct` |
| `brand` | **drop** | AP 0.132 alone vs 0.130 random; removing it is +0.002 AP |
| `country_of_origin` | **drop** | AP 0.133 alone; removing it is −0.000 AP; all 10 levels on the average |
| `package_size` | **drop** | as `package_value`: AP 0.125 alone, *below* random; removing it is +0.001 |
| `dimensions_in` | **drop** | as `volume_in3`: AP 0.133 alone; removing it is −0.002 AP |
| `timestamp` | **drop** | as `month_index`: AP 0.132 alone; removing it is **+0.006 AP**, see below |
| `query_id` | grouping key only | never a feature; keeps a query's rows in one fold |

### Why `timestamp` is not a feature, and not the basis of the split

Buy rate is flat across every complete quarter:

| Quarter | Rows | Buy rate |
|---|---:|---:|
| 2024Q3 | 1,163 | 0.135 |
| 2024Q4 | 1,308 | 0.118 |
| 2025Q1 | 1,183 | 0.132 |
| 2025Q2 | 1,183 | 0.148 |
| 2025Q3 | 1,222 | 0.127 |
| 2025Q4 | 1,342 | 0.129 |
| 2026Q1 | 1,236 | 0.121 |
| 2026Q2 | 1,250 | 0.129 |
| 2026Q3 | 113 | 0.186 |

![Buy rate per quarter with 95% Wilson intervals hugging the dataset average, and a wide interval on the final partial quarter](figures/buy-rate-by-quarter.png)

*Every complete quarter's interval overlaps the 0.1301 average. The last point is the partial quarter, and
its interval is wide enough to swallow the apparent rise.*

2026Q3 sits above the rest at 0.186, but it holds only 113 rows and is a **partial quarter** — the dataset
ends on 2026-07-08. Across the eight complete quarters the range is 0.118–0.148 around a 0.130 base, which
is no drift worth modelling.

Timestamps also do not group rows the way `query_id` does. Within a single query:

- median span (latest − earliest): **488 days**
- median pairwise difference between two rows: **214 days**

![Histogram of the number of days between the first and last event inside one query, spread across more than a year with a median near 488 days](figures/query-time-span.png)

*A `query_id` groups impressions that are more than a year apart. Sorting by time and cutting would split
almost every query down the middle — the exact leak the grouping exists to prevent — in exchange for
guarding against drift the quarter chart says is not there.*

Either measure says the same thing: a `query_id` is not a browsing session, and a chronological split
would cut queries in half while guarding against drift that is not there.

**The split we use** is the one implemented in `src/partitions.py` — a fixed 20% test set and 5-fold
cross-validation over the remaining 80%, both grouped by `query_id` and stratified on `bought`. Effective
proportions per fold: **64% train / 16% validation / 20% test**.

![Stacked bars showing train, validation and test rows for each of the five folds, with the purchase rate per fold annotated and all five rates near 13%](figures/partition-folds.png)

*The split itself. Zero queries appear on both sides of any fold, the purchase rate is within a point of
0.1301 everywhere, and the same 2,002 test rows sit outside all five folds. Whole queries move together, so
the folds are not exactly equal: training rows range from 6,392 to 6,402 and fold 0 — the one the scripts
print — is the 6,401 / 1,597 quoted in §8. `tests/test_partitions.py` asserts the disjointness and the
proportions; `tests/test_eda_evaluation.py` records every index the fold harness hands to a model and
asserts that none of them belongs to the test set.*

---

## Reproducing

Every table and every figure in this document is produced by one of the commands below. Nothing is typed in
by hand.

The charts are stock Matplotlib over numpy and pandas: there is no `plt.style.use` and no `rcParams`
mutation anywhere in the project, so every figure renders with Matplotlib's own defaults and its default
colour cycle. Whatever look they have is set by explicit keyword arguments in `src/eda/charts.py`.

```bash
pip install -r requirements.txt

python -m unittest discover -s tests    # the protocol claims, as assertions
python -m src.eda.run_structure         # §1, §2, §3, §5, §6, §9 — direct counts
python -m src.eda.run_baselines         # §8 — baseline and oracle tables
python -m src.eda.run_interactions      # additive vs crossed features
python -m src.eda.run_pca               # §7 — component ranking
python -m src.eda.run_figures           # §4, §5 — the two headline figures
python -m src.eda.run_variable_evidence # §9 — the per-column evidence, 14 figures
```

`run_pca` writes its scatter only when asked: `python -m src.eda.run_pca --figure docs/figures/pca-text.png`.

| Module | Responsibility |
|---|---|
| `src/partitions.py` | the query-grouped split: fixed test holdout plus 5 inner folds |
| `src/eda/dataset.py` | reads the CSV, derives `price_pct`, the popularity phrase and the missing flag; never reads `cart` |
| `src/eda/features.py` | feature blocks with an explicit `fit(train_indices)` / `transform(indices)` split |
| `src/eda/evaluation.py` | the fold harness and metric definitions, shared with future models |
| `src/eda/curves.py` | response curves, level rates and Wilson intervals |
| `src/eda/variable_evidence.py` | the univariate, leave-one-out and leakage measurements |
| `src/eda/charts.py`, `src/eda/figures.py` | drawing, kept separate from computing |
| `src/eda/run_structure.py` | structural and data-quality counts |
| `src/eda/run_baselines.py` | baseline and oracle tables |
| `src/eda/run_interactions.py` | additive versus crossed feature sets |
| `src/eda/run_pca.py` | bag-of-words PCA and the scatter figure |
| `src/eda/run_figures.py` | the `price_pct` curve and the additivity chart |
| `src/eda/run_variable_evidence.py` | one figure per disposition in §9 |

| Test module | What it pins down |
|---|---|
| `tests/test_eda_dataset.py` | `cart` is never read; every derived field is a function of its own row |
| `tests/test_eda_features.py` | vocabulary, levels, medians and bucket edges come from training rows only; an unseen level is an all-zero row |
| `tests/test_partitions.py` | no `query_id` crosses a fold or the test boundary; the 64/16/20 proportions |
| `tests/test_eda_evaluation.py` | the fold harness never hands a test-set index to a model |

The fixed test set is not scored anywhere in this document. `src.eda.evaluation.evaluate_on_test` exists
for a single final measurement once a configuration has been chosen.

---

## Cierre: el contrato que entrega a Ejercicio 2

Ejercicio 2 (ver `docs/informe-ejercicio-2.md`) recibe de este análisis exactamente seis columnas, congeladas
en `src/model/eda_contract.py` y `parameters-eda.txt`:

```text
Texto:               title + description + ingredients
Categoricas:          category + allergens
Numerica derivada:    price_position
Control diagnostico:  popularity_phrase, no elegible como entrada final
```

`popularity_phrase` es la frase extraída a mano del título que este EDA identificó como la clave casi
determinística del target (tres niveles de BTR muy separados: ~65%, ~2% y exactamente 0%). Se admite
únicamente como control diagnóstico agregado al conjunto completo (las cotas `L0a`/`L0b` de Ejercicio 2);
nunca reemplaza al texto crudo y nunca es finalista. `cart`, los identificadores, el timestamp, los filtros,
el precio absoluto, `storage_type`, `unit_of_measure`, `net_weight_oz` y `nutrition_score` no entran a
ningún candidato de Ejercicio 2.
