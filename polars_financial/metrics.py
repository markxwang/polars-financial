import polars as pl
from polars_financial.days import DAILY


def _get_inv_year(expr: pl.Expr, annual_obs: int | float | None = None):
    """Calculate the number of years based on the annual number of observations."""
    if annual_obs is None:
        return 1
    else:
        return annual_obs / expr.len()


@pl.api.register_expr_namespace("metrics")
class MetricsExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr.fill_nan(None)

    def _add_one_cum_prod(self):
        return self._expr.fill_null(0).add(1).cum_prod()

    def excess_return(self, another_return: pl.Expr | float | int):
        return self._expr - another_return

    def simple_return(self):
        return self._expr.pct_change()

    def cum_return(self):
        return self._add_one_cum_prod() - 1

    def cum_return_final(self):
        return self.cum_return().last()

    def ann_return(self, annual_obs: int | float = DAILY):
        inv_year = _get_inv_year(self._expr, annual_obs)
        return self._add_one_cum_prod().last().pow(inv_year) - 1

    def volatility(self):
        return self._expr.std()

    def ann_volatility(self, annual_obs: int | float = DAILY):
        return self.volatility() * (annual_obs**0.5)

    def sharpe_ratio(self, risk_free: pl.Expr | float | int = 0.0):
        excess_return = self.excess_return(risk_free)
        sr_expr = (
            excess_return.mean() / MetricsExpr(excess_return).volatility()
        ).fill_nan(None)

        return sr_expr

    def ann_sharpe_ratio(
        self, risk_free: pl.Expr | float | int = 0.0, annual_obs: int | float = DAILY
    ):
        return self.sharpe_ratio(risk_free=risk_free) * (annual_obs**0.5)

    def sortino_ratio(self, required_return: pl.Expr | float | int = 0.0):
        sr_expr = self.excess_return(required_return).mean() / self.downside_risk(
            required_return=required_return
        )
        return sr_expr

    def ann_sortino_ratio(
        self,
        required_return: pl.Expr | float | int = 0.0,
        annual_obs: int | float = DAILY,
    ):
        # TODO: check if this is correct
        return self.sortino_ratio(required_return=required_return) * (annual_obs**0.5)

    def downside_risk(self, required_return: pl.Expr | float | int = 0.0):
        dr_expr = (
            self.excess_return(required_return).clip(upper_bound=0).pow(2).mean().sqrt()
        )
        return dr_expr

    def ann_downside_risk(
        self,
        required_return: pl.Expr | float | int = 0.0,
        annual_obs: int | float = DAILY,
    ):
        adr_expr = self.downside_risk(required_return=required_return) * (
            annual_obs**0.5
        )

        return adr_expr

    def information_ratio(self, benchmark: pl.Expr):
        active_return = self.excess_return(benchmark)
        tracking_error = active_return.std()

        ir_expr = active_return.mean() / tracking_error
        return ir_expr

    def max_drawdown(self):
        cum_level = self._add_one_cum_prod()
        cum_max_level = cum_level.cum_max()
        mdd_expr = (cum_level / cum_max_level).min() - 1

        return mdd_expr

    def calmar_ratio(
        self,
        annual_obs: int | float = DAILY,
    ):
        cr_expr = pl.when(self.max_drawdown().ne(0)).then(
            self.ann_return(annual_obs=annual_obs) / self.max_drawdown().abs()
        )
        return cr_expr

    def up_capture_ratio(
        self,
        benchmark: pl.Expr,  # TODO support string/selectors/expr
        annual_obs: int | float = DAILY,
    ):
        up_returns = self._expr.filter(benchmark.fill_nan(None) > 0)
        up_benchmark = benchmark.filter(benchmark.fill_nan(None) > 0)

        return_up = MetricsExpr(up_returns).ann_return(annual_obs=annual_obs)
        benchmark_up = MetricsExpr(up_benchmark).ann_return(annual_obs=annual_obs)

        ucr_expr = return_up / benchmark_up
        return ucr_expr

    def down_capture_ratio(self, benchmark: pl.Expr, annual_obs: int | float = DAILY):
        down_returns = self._expr.filter(benchmark.fill_nan(None) < 0)
        down_benchmark = benchmark.filter(benchmark.fill_nan(None) < 0)

        return_down = MetricsExpr(down_returns).ann_return(annual_obs=annual_obs)
        benchmark_down = MetricsExpr(down_benchmark).ann_return(annual_obs=annual_obs)

        dcr_expr = return_down / benchmark_down
        return dcr_expr

    def alpha_beta(self, benchmark: pl.Expr, risk_free: pl.Expr | float | int = 0.0):
        beta = pl.cov(self._expr, benchmark) / benchmark.var()
        alpha = self._expr - beta * benchmark
        return beta.name.suffix("_beta"), alpha.name.suffix("_alpha")

    def value_at_risk(self, cutoff: float = 0.05):
        return self._expr.quantile(cutoff, interpolation="lower")

    def conditional_value_at_risk(self, cutoff: float = 0.05):
        return self._expr.filter(
            self._expr <= self._expr.quantile(cutoff, interpolation="lower")
        ).mean()
