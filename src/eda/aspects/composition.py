"""Aspect 5: what the product is made of — allergens, ingredients, nutrition score."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eda import report
from src.eda.contribution import holding
from src.eda.plots import bar_by_bucket, bar_by_level
from src.eda.rates import rate_by_bucket, rate_by_level

NO_ALLERGENS = "(no declara)"
"""Level given to the missing values of ``allergens`` so they stay in the table."""

NUTRITION_SENTINEL = 0.0
"""``nutrition_score`` value excluded from the buckets and counted on its own."""

BUCKETS = 8


def analyse(frame: pd.DataFrame, figures: Path) -> dict[str, pd.DataFrame]:
    """Rate the three composition columns, each with the sentinel it carries."""
    report.heading("Aspecto 5 - La composicion del producto")

    nulls = frame.isna().sum()
    print("\nNulos por columna (solo las que tienen):")
    print(nulls[nulls > 0].rename("nulos").to_string())

    declared = frame.assign(allergens=frame["allergens"].fillna(NO_ALLERGENS))
    allergens = rate_by_level(declared, "allergens")
    report.show(allergens, caption="Tasa de compra por alergeno declarado:")
    bar_by_level(
        allergens,
        title="BTR por alergeno declarado",
        path=figures / "05-composicion-alergenos.png",
        target=frame["bought"].to_numpy(),
    )

    dentro = holding(declared, "allergens", "category")
    print("\nSeparacion de allergens dentro de cada categoria:")
    print(dentro.to_string())

    exploded = frame.assign(ingredient=frame["ingredients"].str.split(", ")).explode("ingredient")
    ingredients = rate_by_level(exploded, "ingredient")
    report.show(ingredients, caption="Tasa de compra por ingrediente individual:")
    bar_by_level(
        ingredients,
        title="BTR por ingrediente (una fila cuenta en cada uno de los suyos)",
        path=figures / "05-composicion-ingredientes.png",
    )

    categories_per_recipe = frame.groupby("ingredients")["category"].nunique()
    print()
    report.value("Combinaciones distintas de ingredients", frame["ingredients"].nunique())
    report.value("Ingredientes sueltos distintos", len(ingredients))
    report.value(
        "Categorias por combinacion",
        f"min {categories_per_recipe.min()}, max {categories_per_recipe.max()}",
    )

    sentinel = frame["nutrition_score"] == NUTRITION_SENTINEL
    print()
    report.value(f"Filas con nutrition_score = {NUTRITION_SENTINEL:.0f}", int(sentinel.sum()))
    report.value("Categorias de esas filas", ", ".join(sorted(frame.loc[sentinel, "category"].unique())))
    report.value("BTR de esas filas", f"{frame.loc[sentinel, 'bought'].mean() * 100:.1f}%")

    nutrition = rate_by_bucket(frame[~sentinel], "nutrition_score", buckets=BUCKETS)
    report.show(nutrition, caption="Tasa de compra por tramo de nutrition_score (sin el centinela):")
    bar_by_bucket(
        nutrition,
        title="BTR por tramo de puntaje nutricional (excluye el centinela 0)",
        xlabel="nutrition_score",
        path=figures / "05-composicion-nutricion.png",
        target=frame.loc[~sentinel, "bought"].to_numpy(),
    )

    return {"allergens": allergens, "allergens_dado_category": dentro,
            "ingredients": ingredients, "nutrition_score": nutrition}
