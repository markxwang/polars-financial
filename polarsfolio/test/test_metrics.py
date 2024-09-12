import pytest
import polars as pl
import polarsfolio.metrics  # noqa
from math import isclose
from polars.testing import assert_frame_equal

returns = {
    "empty": [],
    "none": [None, None],
    "one-return": [0.01],
    "simple-benchmark": [0.0, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0],
    "mixed-nan": [float("nan"), 0.01, 0.1, -0.04, 0.02, 0.03, 0.02, 0.01, -0.1],
    "mixed-none": [None, 0.01, 0.1, -0.04, 0.02, 0.03, 0.02, 0.01, -0.1],
    "positive": [0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    "negative": [0.0, -0.06, -0.07, -0.01, -0.09, -0.02, -0.06, -0.08, -0.05],
}


cum_return_data_prep = [
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


@pytest.mark.parametrize("input, expected", cum_return_data_prep)
def test_cum_return(input, expected):

    schema = {"returns": pl.Float64}
    df_output = pl.DataFrame({"returns": input}, schema=schema).select(
        pl.col("returns").metrics.cum_return()
    )
    df_expected = pl.DataFrame({"returns": expected}, schema=schema)

    assert_frame_equal(df_output, df_expected)


cum_return_final_data_prep = [
    # (returns["empty"], None),
    (returns["none"], 0),
    (returns["mixed-nan"], 0.0389311),
    (returns["mixed-none"], 0.038931),
    (returns["negative"], -0.3659073),
]


@pytest.mark.parametrize("input, expected", cum_return_final_data_prep)
def test_cum_return_final(input, expected):
    schema = {"returns": pl.Float64}
    df_output = pl.DataFrame({"returns": input}, schema=schema).select(
        pl.col("returns").metrics.cum_return_final()
    )

    if expected is None:
        assert df_output[0, 0] is None
    else:
        assert isclose(df_output[0, 0], expected, rel_tol=1e-05)


max_drawdown_data_prep = [
    # (returns["empty"], None),
    (returns["none"], 0),
    (returns["one-return"], 0),
    (returns["mixed-nan"], -0.1),
    (returns["positive"], 0),
    (returns["negative"], -0.365907),
]


@pytest.mark.parametrize("input, expected", max_drawdown_data_prep)
def test_max_drawdown(input, expected):
    schema = {"returns": pl.Float64}
    df_output = pl.DataFrame({"returns": input}, schema=schema).select(
        pl.col("returns").metrics.max_drawdown()
    )

    if expected is None:
        assert df_output[0, 0] is None
    else:
        assert isclose(df_output[0, 0], expected, rel_tol=1e-05)
