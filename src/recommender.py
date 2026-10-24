"""Rules based on the user-provided DesiProtein reference. Not ML predictions."""
CROPS = {
 "Horse gram": {"protein":"~22–24% of seed","limits":"Tannins, trypsin inhibitor, phytate, hemagglutinin; described as hard-to-cook.","evidence":"Crop-level literature summary"},
 "Moth bean": {"protein":"~20–26% of seed","limits":"Sulfur amino acids are limiting; low trypsin inhibitor and no hemagglutinin versus soy reported in the reference.","evidence":"Crop-level literature summary"},
 "Rice bean": {"protein":"~17.5–25.6% of seed","limits":"Variety-dependent; tannins, phenolics, trypsin inhibitor, phytate and saponins are reported.","evidence":"Crop-level literature summary"},
 "Lablab": {"protein":"~24–29% in 10 reported cultivars","limits":"Cyanogenic glycosides (especially dark-seeded types), trypsin inhibitor and lectins are concerns.","evidence":"Crop-level literature summary"},
}
PRODUCTS = {
 "Paneer-like curd":"Firm, relatively neutral-tasting curd",
 "Tempeh-style block":"Fermented block",
 "Nuggets / chunks":"Fibrous meat-like structure",
 "Crisps / bhujia-style snack":"Crisp texture and strong seasoning",
}
METHODS = ["Soaking","Germination","Cooking / boiling","Roasting","Fermentation","Protein isolation","Extrusion"]

def profile(crop):
    return CROPS[crop]

def safety_check(crop):
    if crop == "Lablab":
        return ("MANDATORY: mature lablab seeds must be soaked overnight, drained, and boiled in at least "
                "one change of fresh water until soft. Never use raw or lightly heated mature lablab flour. "
                "This app cannot verify a process was actually followed.")
    return "No crop-specific hard rule is triggered here. This is not a food-safety certification."

def recommend(crop, product, method):
    out = []
    if crop == "Horse gram" and method in ("Soaking","Germination"):
        out.append({"title":"Horse gram: soak + germinate",
          "text":"The supplied reference identifies 18 h soaking followed by 48 h germination in light as a literature starting point for experiment planning. It is not guaranteed to be optimal for every variety or outcome.",
          "evidence":"Crop / literature-derived",
          "caveat":"Record variety, exact conditions, assay method and matched control/treatment values before using results as training data."})
    if crop == "Rice bean" and method in ("Soaking","Germination","Cooking / boiling"):
        out.append({"title":"Rice bean: compare processing treatments",
          "text":"The reference reports studies of soaking, sprouting, cooking and autoclaving for antinutritional factors. Compare treatments against matched untreated controls.",
          "evidence":"Crop-specific studies summarized in reference",
          "caveat":"Check units, sample basis and assay methods before combining studies."})
    if crop == "Moth bean" and method in ("Germination","Cooking / boiling"):
        out.append({"title":"Moth bean: sprouting/cooking comparison",
          "text":"The reference identifies research on sprouting and cooking moth bean seed and flour. Extract the exact conditions and measured outcomes; this app does not infer the outcome.",
          "evidence":"Crop-specific studies summarized in reference",
          "caveat":"Keep seed and flour observations distinct."})
    if crop == "Lablab":
        out.append({"title":"Lablab safety gate",
          "text":"Any real-world workflow must require overnight soaking, draining, and boiling in at least one change of fresh water until soft. Do not use raw or lightly heated mature lablab flour.",
          "evidence":"Safety rule from supplied reference",
          "caveat":"A recommendation engine is not a validated food-safety process."})
    product_notes = {
      "Paneer-like curd":"Lablab and moth bean are plausible gel/curd exploration candidates, but the reference does not establish a validated plant-based paneer process for these four crops.",
      "Tempeh-style block":"The reference treats fermented blocks as a promising exploration. Proposed crop blends remain hypotheses.",
      "Nuggets / chunks":"The reference says direct nugget-texture data for these four crops are absent. Do not claim predicted fibrous texture without experiments.",
      "Crisps / bhujia-style snack":"The reference identifies moth bean bhujia and roasted horse gram as plausible directions; extruded crisps from these crops are not established.",
    }
    out.append({"title":"Product-fit note: "+product, "text":product_notes[product],
                "evidence":"Evidence-limited / Test",
                "caveat":"This is a product-development hypothesis, not a measured prediction."})
    return out
