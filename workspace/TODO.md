To-do:
- carryover == 0

Implemented Components:

IC-based feature selection and sequential correlation pruning.
LightGBM forecasting of 60-minute realized volatility.
Inverse-volatility position sizing (“vol-managed strategy”).
Target volatility set to median pre-test RV.
Weight capping between 0.5 and 2.0.
No lookahead via shifted weights and returns.
Standard backtest metrics: Sharpe, Sortino, drawdown, realized vol.
Cumulative return and weight plots for strategy diagnostics.


Future Improvements:

Log-transforming the volatility target to stabilize variance and reduce sensitivity to extreme volatility spikes.
Feature interaction to capture nonlinear relationships such as hour×DST, spread×volume, and order-flow×volatility combinations.
Feature scaling and transformation (log scaling of volume, winsorizing outliers) to reduce skewness and improve model stability.
Model calibration, e.g., mapping predicted volatility to realized volatility via linear correction or quantile mapping.
Horizon-aligned modeling, such as using block-constant (60-minute) weights or features aggregated over the prediction horizon.
More advanced feature selection, including mutual information, clustering, or SHAP-based pruning.
Regime features that capture trend, liquidity, and volatility regimes.
Transaction cost modeling and turnover penalties for a more realistic volatility-managed backtest.
Alternative models such as CatBoost, XGBoost, or sequence models to compare against LightGBM.
