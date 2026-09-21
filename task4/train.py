"""Train rolling-window DCC-GARCH on backward-looking log returns."""

from __future__ import annotations

import argparse
from pathlib import Path

from dcc import load_panel, log_returns, rolling_dcc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window", type=int, default=252, help="estimation window in trading days")
    p.add_argument("--step", type=int, default=21, help="parameter refit frequency in days")
    p.add_argument("--n-assets", type=int, default=100, help="use the first N assets")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "dcc_rolling.npz",
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    panel = load_panel()
    rets = log_returns(panel)
    result = rolling_dcc(
        rets,
        window=args.window,
        step=args.step,
        n_assets=args.n_assets,
        verbose=not args.quiet,
    )
    path = result.save(args.out)
    mean = result.mean_corr()
    print(f"saved {path}")
    print(f"dates {result.dates[0].date()} -> {result.dates[-1].date()}  ({len(result.dates)} days)")
    print(f"assets {len(result.assets)}  refits {len(result.params)}")
    print(f"mean DCC corr  min={mean.min():.4f}  max={mean.max():.4f}  last={mean.iloc[-1]:.4f}")


if __name__ == "__main__":
    main()
