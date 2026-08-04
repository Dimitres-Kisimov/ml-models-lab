"""Documentation-hygiene gate for the model cards.

Every model in this repo must ship a model card, and every card must stay a
well-formed card. These checks use only the standard library so they run
anywhere, offline and deterministically, with no dependency on numpy/torch.

A "model" is any package under ``mllab/`` that exposes a ``python -m mllab.<x>``
entry point (i.e. contains ``__main__.py``). The mapping between a package and
its card is bidirectional, so the suite fails both when a model gains no card
and when a card is left orphaned after a model is renamed or removed.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MLLAB = REPO_ROOT / "mllab"
CARDS_DIR = REPO_ROOT / "docs" / "model_cards"
INDEX = CARDS_DIR / "README.md"
ROLLUP = REPO_ROOT / "docs" / "MODEL_CARDS.md"

# Sections every card must keep. Dropping one would turn a card into a stub, so
# the gate treats their absence as drift.
REQUIRED_SECTIONS = (
    "## Model details",
    "## Intended use",
    "## Training data",
    "## Evaluation",
    "## Limitations and failure modes",
    "## Ethical and deployment considerations",
)


def model_packages():
    """Model package dirs under mllab/ (those with a __main__.py), sorted."""
    return sorted(
        p.name for p in MLLAB.iterdir() if p.is_dir() and (p / "__main__.py").is_file()
    )


def slug(package_name):
    """demand_forecast_net -> demand-forecast-net (the card file stem)."""
    return package_name.replace("_", "-")


def card_stems():
    """Card file stems under docs/model_cards/, excluding the index README."""
    return sorted(p.stem for p in CARDS_DIR.glob("*.md") if p.name != "README.md")


def test_every_model_has_a_card():
    missing = [name for name in model_packages() if not (CARDS_DIR / f"{slug(name)}.md").is_file()]
    assert not missing, f"models with no model card under docs/model_cards/: {missing}"


def test_no_orphan_cards():
    expected = {slug(name) for name in model_packages()}
    orphans = [stem for stem in card_stems() if stem not in expected]
    assert not orphans, f"cards with no matching model package: {orphans}"


def test_discovered_at_least_the_five_models():
    # Guards against the discovery rule silently matching nothing (e.g. a moved
    # directory), which would make the checks above vacuously pass.
    assert len(model_packages()) >= 5


def test_cards_are_well_formed():
    for name in model_packages():
        card = CARDS_DIR / f"{slug(name)}.md"
        if not card.is_file():
            continue  # a missing card is reported by test_every_model_has_a_card
        text = card.read_text(encoding="utf-8")
        assert text.startswith(f"# Model card: {slug(name)}"), (
            f"{slug(name)}.md must open with '# Model card: {slug(name)}'"
        )
        for section in REQUIRED_SECTIONS:
            assert section in text, f"{slug(name)}.md is missing section '{section}'"
        assert "out of scope" in text.lower(), (
            f"{slug(name)}.md must state its out-of-scope / do-not-use boundary"
        )


def test_index_lists_every_card():
    text = INDEX.read_text(encoding="utf-8")
    missing = [name for name in model_packages() if f"{slug(name)}.md" not in text]
    assert not missing, f"docs/model_cards/README.md does not link cards for: {missing}"


def test_rollup_covers_every_model():
    assert ROLLUP.is_file(), "docs/MODEL_CARDS.md roll-up table is missing"
    text = ROLLUP.read_text(encoding="utf-8")
    missing = [name for name in model_packages() if f"model_cards/{slug(name)}.md" not in text]
    assert not missing, f"docs/MODEL_CARDS.md roll-up does not cover: {missing}"
