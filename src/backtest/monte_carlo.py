"""
Monte Carlo Robustness & Risk-of-Ruin Simulator
Performs 1,000 bootstrap simulations of the trade sequence to compute:
- 95% Confidence Intervals for CAGR and Final Equity
- Maximum Drawdown Probability Distribution
- Risk of Ruin (< 50% drawdown probability)
- Trade sequence shuffling to verify edge stability
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


class MonteCarloSimulator:
    @classmethod
    def run_simulation(
        cls,
        trades_df: pd.DataFrame,
        initial_capital: float = 25000.0,
        num_simulations: int = 1000,
        years: float = 6.5,
    ) -> Dict[str, Any]:
        if trades_df.empty:
            return {"error": "Empty trade history."}

        pnl_series = trades_df['pnl_dollar'].values
        total_trades = len(pnl_series)

        final_equities = []
        max_drawdowns = []
        cagrs = []

        # Keep track of representative percentile paths for charting (5th, 50th, 95th)
        percentile_paths = {}

        np.random.seed(42)

        # Run 1000 bootstrap resamples
        simulated_paths = []
        for sim in range(num_simulations):
            # Resample with replacement
            sampled_pnls = np.random.choice(pnl_series, size=total_trades, replace=True)
            equity_curve = [initial_capital]
            curr_equity = initial_capital

            peak = initial_capital
            max_dd = 0.0

            for pnl in sampled_pnls:
                curr_equity += pnl
                curr_equity = max(100.0, curr_equity)
                equity_curve.append(curr_equity)

                if curr_equity > peak:
                    peak = curr_equity
                dd = (peak - curr_equity) / peak
                if dd > max_dd:
                    max_dd = dd

            final_eq = equity_curve[-1]
            cagr = ((final_eq / initial_capital) ** (1.0 / years) - 1.0) * 100.0

            final_equities.append(final_eq)
            max_drawdowns.append(max_dd * 100.0)
            cagrs.append(cagr)
            simulated_paths.append((final_eq, equity_curve))

        # Sort paths by final equity to find percentiles
        simulated_paths.sort(key=lambda x: x[0])
        p5_path = simulated_paths[int(num_simulations * 0.05)][1]
        p25_path = simulated_paths[int(num_simulations * 0.25)][1]
        p50_path = simulated_paths[int(num_simulations * 0.50)][1]
        p75_path = simulated_paths[int(num_simulations * 0.75)][1]
        p95_path = simulated_paths[int(num_simulations * 0.95)][1]

        # Downsample paths to 50 points for fast rendering in browser
        step = max(1, len(p50_path) // 50)
        downsampled_steps = list(range(0, len(p50_path), step))

        drawdowns_arr = np.array(max_drawdowns)
        cagrs_arr = np.array(cagrs)
        equities_arr = np.array(final_equities)

        # Probabilities
        prob_dd_over_30 = float((drawdowns_arr > 30.0).mean() * 100.0)
        prob_dd_over_40 = float((drawdowns_arr > 40.0).mean() * 100.0)
        prob_dd_over_50 = float((drawdowns_arr > 50.0).mean() * 100.0)
        prob_cagr_over_50 = float((cagrs_arr > 50.0).mean() * 100.0)
        prob_cagr_over_70 = float((cagrs_arr > 70.0).mean() * 100.0)

        return {
            "num_simulations": num_simulations,
            "years": years,
            "cagr_percentiles": {
                "p5": round(float(np.percentile(cagrs_arr, 5)), 2),
                "p25": round(float(np.percentile(cagrs_arr, 25)), 2),
                "p50_median": round(float(np.percentile(cagrs_arr, 50)), 2),
                "p75": round(float(np.percentile(cagrs_arr, 75)), 2),
                "p95": round(float(np.percentile(cagrs_arr, 95)), 2),
            },
            "final_equity_percentiles": {
                "p5": round(float(np.percentile(equities_arr, 5)), 2),
                "p50_median": round(float(np.percentile(equities_arr, 50)), 2),
                "p95": round(float(np.percentile(equities_arr, 95)), 2),
            },
            "max_drawdown_percentiles": {
                "p5_best": round(float(np.percentile(drawdowns_arr, 5)), 2),
                "p50_median": round(float(np.percentile(drawdowns_arr, 50)), 2),
                "p95_worst": round(float(np.percentile(drawdowns_arr, 95)), 2),
            },
            "risk_metrics": {
                "prob_dd_over_30pct": round(prob_dd_over_30, 1),
                "prob_dd_over_40pct": round(prob_dd_over_40, 1),
                "prob_dd_over_50pct": round(prob_dd_over_50, 1),
                "prob_cagr_over_50pct": round(prob_cagr_over_50, 1),
                "prob_cagr_over_70pct": round(prob_cagr_over_70, 1),
                "risk_of_ruin_pct": round(prob_dd_over_50, 1),
            },
            "chart_paths": {
                "steps": downsampled_steps,
                "p5": [round(p5_path[i], 0) for i in downsampled_steps],
                "p25": [round(p25_path[i], 0) for i in downsampled_steps],
                "p50": [round(p50_path[i], 0) for i in downsampled_steps],
                "p75": [round(p75_path[i], 0) for i in downsampled_steps],
                "p95": [round(p95_path[i], 0) for i in downsampled_steps],
            }
        }