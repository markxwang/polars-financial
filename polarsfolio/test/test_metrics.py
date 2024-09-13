import pytest
import polars as pl
import polarsfolio.metrics  # noqa
from math import isclose


returns = {
    "empty": [],
    "none": [None, None],
    "one-return": [0.01],
    "simple-benchmark": [0.0, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0],
    "mixed-nan": [float("nan"), 0.01, 0.1, -0.04, 0.02, 0.03, 0.02, 0.01, -0.1],
    "mixed-none": [None, 0.01, 0.1, -0.04, 0.02, 0.03, 0.02, 0.01, -0.1],
    "positive": [0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    "negative": [0.0, -0.06, -0.07, -0.01, -0.09, -0.02, -0.06, -0.08, -0.05],
    "for-annual": [0.0, 0.01, 0.1, -0.04, 0.02, 0.03, 0.02, 0.01, -0.1],
    "flat-line": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
}


cum_return_test_data = [
    (returns["empty"], []),
    (returns["none"], [0, 0]),
    (
        returns["mixed-nan"],
        [
            0,
            0.01,
            0.111,
            0.06656,
            0.0878912,
            0.1205279,
            0.1429385,
            0.1543679,
            0.0389311,
        ],
    ),
    (
        returns["mixed-none"],
        [
            0,
            0.01,
            0.111,
            0.06656,
            0.0878912,
            0.1205279,
            0.1429385,
            0.1543679,
            0.0389311,
        ],
    ),
    (
        returns["negative"],
        [
            0.0,
            -0.06,
            -0.1258,
            -0.134542,
            -0.2124332,
            -0.2281846,
            -0.2744935,
            -0.3325340,
            -0.3659073,
        ],
    ),
]

cum_return_final_test_data = [
    # (returns["empty"], None),
    (returns["none"], 0),
    (returns["mixed-nan"], 0.0389311),
    (returns["mixed-none"], 0.038931),
    (returns["negative"], -0.3659073),
]


max_drawdown_test_data = [
    # (returns["empty"], None),
    (returns["none"], 0),
    (returns["one-return"], 0),
    (returns["mixed-nan"], -0.1),
    (returns["positive"], 0),
    (returns["negative"], -0.365907),
]

ann_return_test_data = [
    (returns["mixed-nan"], "daily", 1.9135925373194231),
    (returns["for-annual"], "weekly", 0.24690830513998208),
    (returns["for-annual"], "monthly", 0.052242061386048144),
]

ann_volatility_test_data = [
    (returns["flat-line"], "daily", 0.0),
    (returns["mixed-nan"], "daily", 0.9136465399704637),
    (returns["for-annual"], "weekly", 0.38851569394870583),
    (returns["for-annual"], "monthly", 0.18663690238892558),
]

calmar_ratio_test_data = [
    (returns["flat-line"], "daily", None),  # TODO: check if this is actually None?
    (returns["one-return"], "daily", None),  # TODO: check if this is actually None?
    (returns["mixed-nan"], "daily", 19.135925373194233),
    (returns["for-annual"], "weekly", 2.4690830513998208),
    (returns["for-annual"], "monthly", 0.52242061386048144),
]

sharpe_ratio_test_data = [
    (returns["empty"], 0.0, None),
    (returns["none"], 0.0, None),
    (returns["one-return"], 0.0, None),
    (returns["mixed-nan"], pl.col("returns"), None),
    (returns["mixed-nan"], 0.0, 1.7238613961706866),
    (returns["positive"], 0.0, 52.915026221291804),
    (returns["negative"], 0.0, -24.406808633910085),
    (returns["flat-line"], 0.0, float("inf")),
    # (returns["mixed-nan"], simple_benchmark, 0.34111411441060574),   # TODO: add additional tests?
]


def _test_single_value(input, expected, method, *args, **kwargs):
    schema = {"returns": pl.Float64}

    metric = getattr(pl.col("returns").metrics, method)
    df_output = pl.DataFrame({"returns": input}, schema=schema).select(
        metric(*args, **kwargs)
    )

    if expected is None:
        assert df_output[0, 0] is None
    else:
        assert isclose(df_output[0, 0], expected, rel_tol=1e-05)


@pytest.mark.parametrize("input, expected", max_drawdown_test_data)
def test_max_drawdown(input, expected):
    _test_single_value(input, expected, "max_drawdown")


@pytest.mark.parametrize("input, expected", cum_return_final_test_data)
def test_cum_return_final(input, expected):
    _test_single_value(input, expected, "cum_return_final")


@pytest.mark.parametrize("input, freq, expected", ann_return_test_data)
def test_ann_return(input, freq, expected):
    _test_single_value(input, expected, "ann_return", freq=freq)


@pytest.mark.parametrize("input, freq, expected", ann_volatility_test_data)
def test_ann_volatility(input, freq, expected):
    _test_single_value(input, expected, "ann_volatility", freq=freq)


@pytest.mark.parametrize("input, freq, expected", calmar_ratio_test_data)
def test_calmar_ratio(input, freq, expected):
    _test_single_value(input, expected, "calmar_ratio", freq=freq)


@pytest.mark.parametrize("input, risk_free, expected", sharpe_ratio_test_data)
def test_ann_sharpe_ratio(input, risk_free, expected):
    _test_single_value(input, expected, "ann_sharpe_ratio", risk_free=risk_free)


# TODO: input data as dictionary or dataframe? Add dataframe builder helper function
