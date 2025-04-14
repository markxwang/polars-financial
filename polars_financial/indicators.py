import polars as pl
from polars._typing import IntoExpr
from polars._utils.parse import parse_into_expression
from polars._utils.wrap import wrap_expr


def _parse_expr(expr: IntoExpr) -> pl.Expr:
    py_expr = parse_into_expression(expr)
    return wrap_expr(py_expr)


def sma(expr: IntoExpr, length: int = 20) -> pl.Expr:
    return _parse_expr(expr).rolling_mean(window_size=length)


def sm_stdev(expr: IntoExpr, length: int = 20) -> pl.Expr:
    return _parse_expr(expr).rolling_std(window_size=length)


def ema(expr: IntoExpr, length: int = 20) -> pl.Expr:
    return _parse_expr(expr).rolling_mean(window_size=length)


def bollinger_bands(
    expr: IntoExpr, length: int = 20, std_dev: float = 2.0
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:

    middle = sma(expr, length)
    deviation = sm_stdev(expr, length) * std_dev
    upper = middle + deviation
    lower = middle - deviation

    return (
        upper.name.suffix("_bb_up"),
        middle.name.suffix("_bb_mid"),
        lower.name.suffix("_bb_low"),
    )


def macd(
    expr: IntoExpr,
    fast_length: int = 12,
    slow_length: int = 26,
    signal_length: int = 9,
) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:

    fast_ema = ema(expr, fast_length)
    slow_ema = ema(expr, slow_length)

    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_length)

    return (
        fast_ema.name.suffix("_macd_fast"),
        slow_ema.name.suffix("_macd_slow"),
        macd_line.name.suffix("_macd_line"),
        signal_line.name.suffix("_macd_signal"),
    )


def rsi(expr: IntoExpr, length: int = 14) -> pl.Expr:

    delta = _parse_expr(expr).diff()
    gain = delta.clip(upper_bound=0)
    loss = (-delta).clip(upper_bound=0)

    avg_gain = gain.rolling_mean(window_size=length)
    avg_loss = loss.rolling_mean(window_size=length)

    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)

    return rsi
