import polars as pl


@pl.api.register_expr_namespace("metrics")
class MetricsExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def _add_one_cum_prod(self):
        return self._expr.add(1).cum_prod()

    def simple_return(self):
        return self._expr.pct_change()

    def cum_return(self):
        return self._add_one_cum_prod() - 1

    def ann_return(self, num_year: int | float):
        return self._add_one_cum_prod().pow(1 / num_year) - 1

    def ann_volatility(self, num_periods: int | float):
        return self._expr.std() * (num_periods**0.5)

    def sharpe_ratio(self, risk_free: float = 0.0):
        return (self._expr - risk_free) / self._expr.std()

    def sharpe_ratio_ann(
        self,
        num_periods: int | float,
        risk_free: float = 0.0,
    ):
        return (self._expr - risk_free).mul(num_periods**0.5) / self._expr.std()

    def sortino_ratio(self, required_return: float = 0.0):
        excess_returns = self._expr - required_return

        return excess_returns.mean() / self.downside_risk(
            required_return=required_return
        )

    def downside_risk(self, required_return: float = 0.0):
        excess_returns = self._expr - required_return
        downside_returns = excess_returns.clip(upper_bound=0)
        return downside_returns.pow(2).mean().sqrt()

    def information_ratio(self, benchmark: pl.Expr):
        active_return = self._expr - benchmark
        tracking_error = active_return.std()
        return active_return.mean() / tracking_error

    def max_drawdown(self):
        cum_level = self._expr._add_one_cum_prod()
        running_cum_max_level = cum_level.cum_max()
        mdd = (cum_level - running_cum_max_level) / running_cum_max_level

        return mdd.min()

    def calmar_ratio(self, num_year: int | float):
        return self._expr.ann_return(num_year=num_year) / self._expr.max_drawdown()

    def capture_ratio(self, benchmark: pl.Expr, num_year: int | float):
        return self._expr.ann_return(num_year) / benchmark.ann_return(num_year)

    def up_capture_ratio(self, benchmark: pl.Expr, num_year: int | float):
        return_up = self._expr.filter(benchmark >= 0).ann_return(num_year)
        benchmark_up = benchmark.filter(benchmark >= 0).ann_return(num_year)
        return return_up / benchmark_up

    def down_capture_ratio(self, benchmark: pl.Expr, num_year: int | float):
        return_down = self._expr.filter(benchmark <= 0).ann_return(num_year)
        benchmark_down = benchmark.filter(benchmark <= 0).ann_return(num_year)
        return return_down / benchmark_down

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
