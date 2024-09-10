import polars as pl


@pl.api.register_expr_namespace("metrics")
class MetricsExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def _add_one_cum_prod(self):
        return self._expr.add(1).cum_prod()

    def cum_return(self):
        return self._add_one_cum_prod() - 1

    def comp_ann_growth_rate(self, num_year: int | float):
        return self._add_one_cum_prod().pow(1 / num_year) - 1

    def ann_volatility(self, num_periods: int | float):
        return self._expr.std() * (num_periods**0.5)

    def sharpe_ratio(self, risk_free: float = 0.0):
        return (self._expr - risk_free) / self._expr.std()

    def sortino_ratio(self, required_return: float = 0.0):
        excess_returns = self._expr - required_return
        downside_returns = excess_returns.clip(upper_bound=0)
        downside_deviation = downside_returns.pow().mean().sqrt()

        return excess_returns.mean() / downside_deviation

    def information_ratio(self, benchmark: pl.Expr):
        active_return = self._expr - benchmark
        tracking_error = active_return.std()
        return active_return.mean() / tracking_error

    def max_drawdown(self):
        cum_level = self._expr._add_one_cum_prod()
        running_cum_max_level = cum_level.cum_max()
        mdd = (cum_level - running_cum_max_level) / running_cum_max_level

        return mdd.min()

    def alpha_beta(self, benchmark: pl.Expr, risk_free: float = 0.0):
        beta = pl.cov(self._expr, benchmark) / benchmark.var()
        alpha = self._expr - beta * benchmark
        return beta.name.suffix("_beta"), alpha.name.suffix("_alpha")
