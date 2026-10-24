from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = BASE_DIR / "data" / "reference"


def calculate_recipe_protein(recipe_id: str) -> dict:
    """Calculate protein from ingredient composition and recipe quantities."""

    ingredients = pd.read_csv(
        REFERENCE_DIR / "ingredients.csv"
    )
    recipes = pd.read_csv(
        REFERENCE_DIR / "recipe_ingredients.csv"
    )

    recipe = recipes[recipes["recipe_id"] == recipe_id].copy()

    if recipe.empty:
        raise ValueError(f"Recipe not found: {recipe_id}")

    recipe = recipe.merge(
        ingredients,
        on="ingredient",
        how="left",
        validate="many_to_one",
    )

    if recipe["protein_g_per_100g"].isna().any():
        missing = recipe.loc[
            recipe["protein_g_per_100g"].isna(),
            "ingredient",
        ].tolist()
        raise ValueError(f"Missing protein data for: {missing}")

    total_mass_g = recipe["quantity_g"].sum()

    recipe["protein_g"] = (
        recipe["quantity_g"]
        * recipe["protein_g_per_100g"]
        / 100
    )

    total_protein_g = recipe["protein_g"].sum()

    return {
        "recipe_id": recipe_id,
        "total_mass_g": float(total_mass_g),
        "total_protein_g": round(float(total_protein_g), 2),
        "protein_percent_dry_mix": round(
            float(total_protein_g / total_mass_g * 100), 2
        ),
        "ingredient_breakdown": recipe[
            ["ingredient", "quantity_g", "protein_g"]
        ].to_dict(orient="records"),
    }



def analyze_legume_amino_acids(recipe_id: str = "patty_001") -> dict:
    """Analyze the amino-acid profile of the legume portion of a recipe.

    This excludes non-legume protein sources from the amino-acid
    calculation, so it is not the complete recipe profile.
    """
    amino_path = REFERENCE_DIR / "amino_acid_profiles.csv"
    amino = pd.read_csv(amino_path)
    amino.columns = amino.columns.str.strip()

    ingredients = pd.read_csv(REFERENCE_DIR / "ingredients.csv")
    recipes = pd.read_csv(REFERENCE_DIR / "recipe_ingredients.csv")

    recipe = recipes[recipes["recipe_id"] == recipe_id].merge(
        ingredients,
        on="ingredient",
        how="left",
        validate="many_to_one",
    )

    if recipe.empty:
        raise ValueError(f"Recipe not found: {recipe_id}")

    # Map recipe ingredient names to the crop names in the amino-acid CSV.
    legume_map = {
        "Rice Bean Flour": "Rice Bean",
        "Moth Bean Flour": "Moth Bean",
        "Red Lentil Flour": "Red Lentil",
    }

    recipe["crop"] = recipe["ingredient"].map(legume_map)
    legumes = recipe[recipe["crop"].notna()].copy()

    if legumes.empty:
        raise ValueError("No supported legumes found in this recipe.")

    if legumes["protein_g_per_100g"].isna().any():
        raise ValueError("Missing protein data for a legume.")

    # Protein contributed by each legume, in grams.
    legumes["protein_g"] = (
        legumes["quantity_g"] * legumes["protein_g_per_100g"] / 100
    )
    protein_by_crop = legumes.groupby("crop")["protein_g"].sum()
    total_legume_protein = float(protein_by_crop.sum())

    if total_legume_protein <= 0:
        raise ValueError("Total legume protein must be positive.")

    required = {
        "Amino Acid",
        "Rice Bean",
        "Moth Bean",
        "Red Lentil",
        "WHO Adult Requirement",
    }
    missing = required - set(amino.columns)
    if missing:
        raise ValueError(
            f"Missing columns in amino-acid CSV: {sorted(missing)}"
        )

    results = []
    for _, row in amino.iterrows():
        weighted = 0.0

        for crop in ["Rice Bean", "Moth Bean", "Red Lentil"]:
            crop_protein = float(protein_by_crop.get(crop, 0.0))
            concentration = pd.to_numeric(
                pd.Series([row[crop]]), errors="coerce"
            ).iloc[0]

            if pd.notna(concentration):
                weighted += crop_protein * float(concentration)
            elif crop_protein > 0:
                raise ValueError(
                    f"Missing {row['Amino Acid']} data for {crop}."
                )

        weighted /= total_legume_protein

        requirement = pd.to_numeric(
            pd.Series([row["WHO Adult Requirement"]]),
            errors="coerce",
        ).iloc[0]

        score = (
            weighted / requirement * 100
            if pd.notna(requirement) and requirement > 0
            else None
        )

        results.append({
            "amino_acid": row["Amino Acid"],
            "g_per_100g_protein": round(weighted, 2),
            "reference_value": (
                float(requirement) if pd.notna(requirement) else None
            ),
            "percent_of_reference": (
                round(score, 1) if score is not None else None
            ),
        })

    return {
        "recipe_id": recipe_id,
        "total_legume_protein_g": round(total_legume_protein, 2),
        "amino_acids": results,
        "scope": (
            "Legume portion only; excludes wheat gluten, soy protein, "
            "chickpea flour, and other ingredients."
        ),
    }




if __name__ == "__main__":
    result = calculate_recipe_protein("patty_001")

    print(f"Recipe: {result['recipe_id']}")
    print(f"Dry mix mass: {result['total_mass_g']:.1f} g")
    print(f"Protein: {result['total_protein_g']:.2f} g")
    print(
        "Protein percentage: "
        f"{result['protein_percent_dry_mix']:.2f}%"
    )

    print("\nIngredient breakdown:")
    for item in result["ingredient_breakdown"]:
        print(
            f"{item['ingredient']}: "
            f"{item['protein_g']:.2f} g protein"
        )

    print("\nThree-legume amino-acid analysis:")
    amino_result = analyze_legume_amino_acids()
    print("Scope:", amino_result["scope"])
    print(
        "Legume protein:",
        amino_result["total_legume_protein_g"],
        "g",
    )

    for item in amino_result["amino_acids"]:
        reference = item["percent_of_reference"]
        reference_text = (
            f"{reference}%"
            if reference is not None
            else "Not comparable"
        )
        print(
            f"{item['amino_acid']}: "
            f"{item['g_per_100g_protein']} g/100g protein | "
            f"Reference comparison: {reference_text}"
        )
