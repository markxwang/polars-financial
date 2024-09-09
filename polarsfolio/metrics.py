import polars as pl


@pl.api.register_expr_namespace("metrics")
class MetricsExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def max_drawdown(self):
        cum_return = self._expr.add(1).cum_prod()
        running_cum_max_return = cum_return.cum_max()
        mdd = (cum_return - running_cum_max_return) / running_cum_max_return

        return mdd.min()

    def alpha_beta(self, benchmark: pl.Expr, risk_free: float = 0.0):
        beta = pl.cov(self._expr, benchmark) / benchmark.var()
        alpha = self._expr - beta * benchmark
        return beta.name.suffix("_beta"), alpha.name.suffix("_alpha")
