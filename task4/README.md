Here, I am going to work on the GCN/GNN part to get more insight between how the stocks are moving

I will compare my findings against:
DCC metric


Todos:
[x] Model DCC
[] Model GNN
[] Check dif
[] Innovate

## DCC (rolling window)

Engle (2002) DCC-GARCH, estimated on **backward-looking** windows only.

- Univariate GARCH(1,1) per asset, then DCC(1,1) on standardized residuals
- At date `t`, parameters are fit on returns in `[t - window, t)`
- `R_t` is the one-step-ahead correlation (uses data through `t-1`)
- Parameters are refit every `step` days; between refits the filter walks forward with frozen params

```
python train.py --window 252 --step 21 --n-assets 100
```

Outputs land in `artifacts/`:
- `dcc_rolling.npz` — one-step-ahead `R_t` for each date
- `dcc_rolling_params.csv` — rolling `(a, b)` and GARCH means
- `dcc_rolling_mean_corr.csv` — mean off-diagonal DCC correlation
