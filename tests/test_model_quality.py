"""The model has to clear the only baseline that matters."""

from src.predict import (
    confidence_tiers,
    home_court_baseline,
    score,
)

# Always picking the home team is 56.8% accurate on this data. A model that
# cannot beat it has not learned anything worth having.
HOME_COURT_BASELINE = 0.568


def test_beats_home_court_baseline(reconciled):
    accuracy = score(reconciled)["accuracy"]
    assert accuracy > HOME_COURT_BASELINE, (
        f"accuracy {accuracy:.4f} does not beat the home-court baseline"
    )


def test_home_court_baseline_is_where_we_think_it_is(predictions):
    measured = home_court_baseline(predictions)
    assert 0.55 < measured < 0.59, f"home-court baseline moved: {measured:.4f}"


def test_confident_tier_beats_toss_up_tier(reconciled):
    """The headline result: separating by confidence has to actually separate."""
    tiers = confidence_tiers(reconciled).set_index("tier")
    toss_up, _, confident = (tiers.iloc[i] for i in range(3))

    assert toss_up["n"] > 0 and confident["n"] > 0
    assert confident["accuracy"] > toss_up["accuracy"] + 0.10


def test_tiers_partition_the_predictions(reconciled):
    tiers = confidence_tiers(reconciled)
    assert tiers["n"].sum() == len(reconciled)
