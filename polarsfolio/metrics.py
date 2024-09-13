import polars as pl
from typing import Literal

FREQ_TO_PERIODS = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
}

freq_type = Literal["daily", "weekly", "monthly"]


def _get_sqrt_periods(freq: freq_type) -> float:
    return FREQ_TO_PERIODS[freq] ** 0.5


def _get_year(freq: freq_type):
    return pl.len() / FREQ_TO_PERIODS[freq]


@pl.api.register_expr_namespace("metrics")
class MetricsExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr.fill_nan(None)

    def _add_one_cum_prod(self):
        return self._expr.fill_null(0).add(1).cum_prod()

    def _excess_return(self, another_return: pl.Expr):
        return self._expr - another_return

    def _mean_excess_return(self, another_return: pl.Expr):
        return self._excess_return(another_return).mean()

    def simple_return(self):
        return self._expr.pct_change()

    def cum_return(self):
        return self._add_one_cum_prod() - 1

    def cum_return_final(self):
        return self.cum_return().last()

    def ann_return(self, freq: freq_type):
        year = _get_year(freq)
        return self._add_one_cum_prod().last().pow(1 / year) - 1

    def volatility(self):
        return self._expr.std()

    def ann_volatility(self, freq: freq_type = "daily"):
        return self.volatility() * _get_sqrt_periods(freq)

    def sharpe_ratio(self, risk_free: float = 0.0):
        sr_expr = self._mean_excess_return(risk_free) / self.volatility()
        return sr_expr

    def ann_sharpe_ratio(self, risk_free: float = 0.0, freq: freq_type = "daily"):
        return self.sharpe_ratio(risk_free=risk_free) * _get_sqrt_periods(freq)

    def sortino_ratio(self, required_return: float = 0.0):
        sr_expr = self._mean_excess_return(required_return) / self.downside_risk(
            required_return=required_return
        )
        return sr_expr

    def ann_sortino_ratio(
        self, required_return: float = 0.0, freq: freq_type = "daily"
    ):
        # TODO: check if this is correct
        return self.sortino_ratio(required_return=required_return) * _get_sqrt_periods(
            freq
        )

    def downside_risk(self, required_return: float = 0.0):
        dr_expr = (
            self._excess_return(required_return)
            .clip(upper_bound=0)
            .pow(2)
            .mean()
            .sqrt()
        )
        return dr_expr

    def ann_downside_risk(
        self, required_return: float = 0.0, freq: freq_type = "daily"
    ):
        adr_expr = self.downside_risk(
            required_return=required_return
        ) * _get_sqrt_periods(freq)

        return adr_expr

    def information_ratio(self, benchmark: pl.Expr):
        active_return = self._excess_return(benchmark)
        tracking_error = active_return.std()

        ir_expr = active_return.mean() / tracking_error
        return ir_expr

    def max_drawdown(self):
        cum_level = self._add_one_cum_prod()
        cum_max_level = cum_level.cum_max()
        mdd_expr = (cum_level / cum_max_level).min() - 1

        return mdd_expr

    def calmar_ratio(self, freq: freq_type = "daily"):
        cr_expr = pl.when(self.max_drawdown().ne(0)).then(
            self.ann_return(freq=freq) / self.max_drawdown().abs()
        )
        return cr_expr

    def up_capture_ratio(self, benchmark: pl.Expr, freq: freq_type = "daily"):
        up_returns = self._expr.filter(benchmark >= 0)
        up_benchmark = benchmark.filter(benchmark >= 0)

        return_up = MetricsExpr(up_returns).ann_return(freq=freq)
        benchmark_up = MetricsExpr(up_benchmark).ann_return(freq=freq)

        ucr_expr = return_up / benchmark_up
        return ucr_expr

    def down_capture_ratio(self, benchmark: pl.Expr, freq: freq_type = "daily"):
        down_returns = self._expr.filter(benchmark < 0)
        down_benchmark = benchmark.filter(benchmark < 0)

        return_down = MetricsExpr(down_returns).ann_return(freq=freq)
        benchmark_down = MetricsExpr(down_benchmark).ann_return(freq=freq)

        dcr_expr = return_down / benchmark_down
        return dcr_expr

    def alpha_beta(self, benchmark: pl.Expr, risk_free: float = 0.0):
        beta = pl.cov(self._expr, benchmark) / benchmark.var()
        alpha = self._expr - beta * benchmark
        return beta.name.suffix("_beta"), alpha.name.suffix("_alpha")

    def value_at_risk(self, cutoff: float = 0.05):
        return self._expr.quantile(cutoff, interpolation="lower")

    def conditional_value_at_risk(self, cutoff: float = 0.05):
        return self._expr.filter(
            self._expr <= self._expr.quantile(cutoff, interpolation="lower")
        ).mean()
