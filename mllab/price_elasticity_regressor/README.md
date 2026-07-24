# price-elasticity-regressor

Per-segment log-log price elasticity with ridge and lasso implemented from
scratch, turned into a pricing decision via the Lerner optimal-markup rule -
and an explicit endogeneity caveat, because in pricing data accuracy is not the
same as good decisions.

- **Build:** ridge (closed form via `scipy.linalg.solve` as the oracle, plus
  gradient descent from scratch that a test shows converges to it) and lasso
  (coordinate descent with soft-thresholding; a test shows strong penalties zero
  out coefficients). Constant-elasticity demand `ln Q = a + b ln P`; optimal
  price `P* = MC*|e|/(|e|-1)`, guarded so `|e| <= 1` returns the cap (no interior
  optimum).
- **Improvement:** hierarchical shrinkage (partial pooling) of each segment's
  elasticity toward the pooled estimate, which helps thin segments; and an
  explicit endogeneity demonstration - the synthetic price is set partly from a
  latent demand shock, so naive OLS is biased and adding the control recovers
  the true elasticity.
- **Data:** ~60 products x 50-300 transactions with a known true elasticity and
  a baked-in demand confounder (`mllab/synth.make_transactions`).
- **Metric:** elasticity RMSE / R^2 **and** decision quality - simulated profit
  regret vs the analytic optimum and uplift vs a cost-plus policy.

Measured (seed 11): naive-OLS elasticity bias +1.52 (RMSE 1.58) collapses to
+0.03 (RMSE 0.16) once the confounder is controlled, and shrinkage tightens RMSE
to 0.13; mean profit regret is 0.89% vs the oracle price, +18.7% uplift vs
cost-plus.

## Sources

- Ridge regression - Hoerl & Kennard, *Technometrics* 1970 - https://doi.org/10.1080/00401706.1970.10488634
- Lasso - Tibshirani, *JRSS-B* 1996 - https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
- Coordinate descent for the lasso - Friedman, Hastie & Tibshirani, 2010 - https://doi.org/10.18637/jss.v033.i01
- Lerner index / optimal markup - Lerner, *Rev. Econ. Studies* 1934 - https://doi.org/10.2307/2967480
