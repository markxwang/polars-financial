import polars as pl
from polars._typing import IntoExpr

from .indicators import sma


def sma_crossover(
    expr: IntoExpr,
    fast_length: int = 20,
    slow_length: int = 50,
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    fast_sma = sma(expr, fast_length)
    slow_sma = sma(expr, slow_length)

    prev_fast = fast_sma.shift(1)
    prev_slow = slow_sma.shift(1)

    crossover = (
        pl.when((fast_sma > slow_sma) & (prev_fast <= prev_slow))
        .then(1)
        .when((fast_sma < slow_sma) & (prev_fast >= prev_slow))
        .then(-1)
        .otherwise(0)
    )

    return (
        fast_sma.name.suffix("_fast_sma"),
        slow_sma.name.suffix("_slow_sma"),
        crossover.name.suffix("_crossover"),
    )
