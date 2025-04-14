import polars as pl
from .indicators import sma, sm_stdev, ema, bollinger_bands, macd, rsi


@pl.api.register_expr_namespace("ta")
class TechnicalAnalysis:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def sma(self, length: int = 20) -> pl.Expr:
        return sma(self._expr, length)

    def sm_stdev(self, length: int = 20) -> pl.Expr:
        return sm_stdev(self._expr, length)

    def ema(self, length: int = 20) -> pl.Expr:
        return ema(self._expr, length)

    def bollinger_bands(self, length: int = 20, std_dev: float = 2.0) -> pl.Expr:
        return bollinger_bands(self._expr, length, std_dev)

    def macd(
        self, fast_length: int = 12, slow_length: int = 26, signal_length: int = 9
    ) -> pl.Expr:
        return macd(self._expr, fast_length, slow_length, signal_length)

    def rsi(self, length: int = 14) -> pl.Expr:
        return rsi(self._expr, length)
