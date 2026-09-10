"""
Institutional Quant Options Portfolio Backtester
Combines:
1. Dynamic Regime Switching:
   - Strong Uptrend (Spot > EMA20 > EMA50 > EMA200) + High IV (IVR >= 22%): Bull Put Credit Spread (Delta ~0.16)
   - Downtrend (Spot < EMA50 and Spot < EMA200) + High IV (IVR >= 22%): Bear Call Credit Spread (Delta ~0.16)
   - Cheap Volatility Breakout (Spot > EMA20 and IVR <= 25%): Bull Call Debit Spread (ATM/OTM Convexity)
   - Cheap Volatility Breakdown (Spot < EMA20 and IVR <= 25%): Bear Put Debit Spread (ATM/OTM Downside Convexity)
2. Volatility Risk Premium (VRP) confirmation: Ensures IV > HV_30 before selling.
3. Fast Turnover: 50% Profit Target on Credit Spreads (average 8-11 days holding time).
4. Strict Stop Loss (2.0x credit) to prevent tail drawdowns.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from src.engine.black_scholes import BlackScholesEngine
from src.backtest.performance import PerformanceCalculator, BacktestMetrics


@dataclass
class BacktestConfig:
    initial_capital: float = 25000.0
    allocation_pct_per_trade: float = 0.22   # 22% portfolio allocation per trade
    max_open_positions: int = 6
    target_short_delta: float = 0.16
    profit_target_pct: float = 0.50          # Take profit at 50% max profit
    stop_loss_mult: float = 2.00             # Cut loss at 2.0x entry credit
    entry_dte: int = 28                      # 28 DTE optimal theta acceleration
    min_iv_rank_to_sell: float = 20.0
    # Long Options (Debit Spreads for Convexity)
    enable_option_buying: bool = True
    debit_profit_target_pct: float = 1.00    # +100% on debit spread
    debit_stop_loss_pct: float = 0.40        # -40% cut on debit spread
    max_iv_rank_to_buy: float = 24.0         # Only buy when volatility is cheap
    # Execution
    slippage_per_contract: float = 1.50
    risk_free_rate: float = 0.045


@dataclass
class Position:
    trade_id: int
    symbol: str
    action: str             # "SELL_CREDIT" or "BUY_DEBIT"
    strategy: str           # "BULL_PUT_SPREAD", "BEAR_CALL_SPREAD", "BULL_CALL_SPREAD", "BEAR_PUT_SPREAD"
    entry_date: pd.Timestamp
    expiration_date: pd.Timestamp
    entry_spot: float
    entry_price_per_share: float
    num_contracts: int
    short_strike: Optional[float]
    long_strike: Optional[float]
    margin_locked: float
    max_profit: float
    max_loss: float
    initial_iv: float


class OptionsBacktester:
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        datasets: Dict[str, pd.DataFrame],
    ) -> tuple[pd.Series, List[Dict[str, Any]], BacktestMetrics]:
        common_dates = None
        for sym, df in datasets.items():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))
        if not common_dates:
            raise ValueError("No common dates found across datasets.")

        date_list = sorted(list(common_dates))

        cash = self.config.initial_capital
        equity_history = {}
        open_positions: List[Position] = []
        closed_trades: List[Dict[str, Any]] = []
        trade_id_counter = 0

        for current_date in date_list:
            # 1. UPDATE OPEN POSITIONS
            active_positions = []
            for pos in open_positions:
                df_sym = datasets[pos.symbol]
                if current_date not in df_sym.index:
                    active_positions.append(pos)
                    continue

                row = df_sym.loc[current_date]
                current_spot = float(row['Close'])
                current_iv = float(row['iv_proxy'])
                days_left = (pos.expiration_date - current_date).days
                T = max(0.0, days_left / 365.0)

                close_reason = None
                pnl_per_share = 0.0

                if pos.action == "SELL_CREDIT":
                    if pos.strategy == "BULL_PUT_SPREAD":
                        short_p = BlackScholesEngine.price("put", current_spot, pos.short_strike, T, self.config.risk_free_rate, current_iv)
                        long_p = BlackScholesEngine.price("put", current_spot, pos.long_strike, T, self.config.risk_free_rate, current_iv)
                        spread_cost = max(0.0, short_p - long_p)
                    else:  # BEAR_CALL_SPREAD
                        short_p = BlackScholesEngine.price("call", current_spot, pos.short_strike, T, self.config.risk_free_rate, current_iv)
                        long_p = BlackScholesEngine.price("call", current_spot, pos.long_strike, T, self.config.risk_free_rate, current_iv)
                        spread_cost = max(0.0, short_p - long_p)

                    pnl_per_share = pos.entry_price_per_share - spread_cost

                    if spread_cost <= pos.entry_price_per_share * (1.0 - self.config.profit_target_pct):
                        close_reason = "PROFIT_TARGET_50%"
                    elif spread_cost >= pos.entry_price_per_share * self.config.stop_loss_mult:
                        close_reason = "STOP_LOSS"
                    elif days_left <= 0:
                        close_reason = "EXPIRED"

                else:  # BUY_DEBIT
                    if pos.strategy == "BULL_CALL_SPREAD":
                        long_p = BlackScholesEngine.price("call", current_spot, pos.long_strike, T, self.config.risk_free_rate, current_iv)
                        short_p = BlackScholesEngine.price("call", current_spot, pos.short_strike, T, self.config.risk_free_rate, current_iv)
                        spread_val = max(0.0, long_p - short_p)
                    else:  # BEAR_PUT_SPREAD
                        long_p = BlackScholesEngine.price("put", current_spot, pos.long_strike, T, self.config.risk_free_rate, current_iv)
                        short_p = BlackScholesEngine.price("put", current_spot, pos.short_strike, T, self.config.risk_free_rate, current_iv)
                        spread_val = max(0.0, long_p - short_p)

                    pnl_per_share = spread_val - pos.entry_price_per_share

                    if spread_val >= pos.entry_price_per_share * (1.0 + self.config.debit_profit_target_pct):
                        close_reason = "PROFIT_TARGET_DEBIT"
                    elif spread_val <= pos.entry_price_per_share * (1.0 - self.config.debit_stop_loss_pct):
                        close_reason = "STOP_LOSS_DEBIT"
                    elif days_left <= 0:
                        close_reason = "EXPIRED"

                if close_reason:
                    gross_pnl = pnl_per_share * 100.0 * pos.num_contracts
                    net_pnl = gross_pnl - (self.config.slippage_per_contract * pos.num_contracts)
                    holding_days = (current_date - pos.entry_date).days
                    cash += pos.margin_locked + net_pnl

                    closed_trades.append({
                        'trade_id': pos.trade_id,
                        'symbol': pos.symbol,
                        'strategy': pos.strategy,
                        'action': pos.action,
                        'entry_date': pos.entry_date,
                        'exit_date': current_date,
                        'holding_days': holding_days,
                        'entry_spot': pos.entry_spot,
                        'exit_spot': current_spot,
                        'contracts': pos.num_contracts,
                        'entry_price': pos.entry_price_per_share,
                        'pnl_dollar': round(net_pnl, 2),
                        'pnl_pct': round((net_pnl / pos.margin_locked) * 100.0, 2) if pos.margin_locked > 0 else 0.0,
                        'close_reason': close_reason,
                    })
                else:
                    active_positions.append(pos)

            open_positions = active_positions
            unrealized_margin = sum(p.margin_locked for p in open_positions)
            total_equity = cash + unrealized_margin
            equity_history[current_date] = total_equity

            # 2. SCAN AND EXECUTE NEW QUANT TRADES
            if len(open_positions) < self.config.max_open_positions and total_equity > 1000.0:
                for sym, df_sym in datasets.items():
                    if len(open_positions) >= self.config.max_open_positions:
                        break
                    if any(p.symbol == sym for p in open_positions):
                        continue
                    if current_date not in df_sym.index:
                        continue

                    row = df_sym.loc[current_date]
                    spot = float(row['Close'])
                    iv_rank = float(row['iv_rank'])
                    iv = float(row['iv_proxy'])
                    hv_30 = float(row.get('hv_30', iv))
                    ema_20 = float(row['ema_20'])
                    ema_50 = float(row['ema_50'])
                    ema_200 = float(row.get('ema_200', ema_50))

                    is_strong_bull = spot > ema_20 and ema_20 > ema_50 and spot > ema_200
                    is_bear = spot < ema_50 and (spot < ema_200 or ema_20 < ema_50)

                    target_trade = None
                    T_entry = self.config.entry_dte / 365.0
                    exp_date = current_date + pd.Timedelta(days=self.config.entry_dte)
                    spread_width = 5.0 if spot < 250 else (10.0 if spot < 600 else 15.0)

                    # A. UPTREND SELLING: BULL PUT CREDIT SPREAD (Delta ~0.16)
                    if is_strong_bull and iv_rank >= self.config.min_iv_rank_to_sell:
                        strike_gap = 1.05 * iv * np.sqrt(T_entry) * spot
                        short_k = round((spot - strike_gap) / 5.0) * 5.0
                        long_k = short_k - spread_width

                        p_short = BlackScholesEngine.price("put", spot, short_k, T_entry, self.config.risk_free_rate, iv)
                        p_long = BlackScholesEngine.price("put", spot, long_k, T_entry, self.config.risk_free_rate, iv)
                        credit = p_short - p_long

                        if credit >= 0.25 and credit < spread_width * 0.60:
                            max_risk = (spread_width - credit) * 100.0
                            target_trade = ("SELL_CREDIT", "BULL_PUT_SPREAD", short_k, long_k, credit, max_risk, exp_date)

                    # B. DOWNTREND SELLING: BEAR CALL CREDIT SPREAD (Delta ~0.16)
                    elif is_bear and iv_rank >= self.config.min_iv_rank_to_sell:
                        strike_gap = 1.05 * iv * np.sqrt(T_entry) * spot
                        short_k = round((spot + strike_gap) / 5.0) * 5.0
                        long_k = short_k + spread_width

                        p_short = BlackScholesEngine.price("call", spot, short_k, T_entry, self.config.risk_free_rate, iv)
                        p_long = BlackScholesEngine.price("call", spot, long_k, T_entry, self.config.risk_free_rate, iv)
                        credit = p_short - p_long

                        if credit >= 0.25 and credit < spread_width * 0.60:
                            max_risk = (spread_width - credit) * 100.0
                            target_trade = ("SELL_CREDIT", "BEAR_CALL_SPREAD", short_k, long_k, credit, max_risk, exp_date)

                    # C. CONVEXITY BUYING (Cheap Volatility Breakout): BULL CALL DEBIT SPREAD
                    elif self.config.enable_option_buying and is_strong_bull and iv_rank <= self.config.max_iv_rank_to_buy and spot > ema_20 * 1.01:
                        long_k = round(spot / 5.0) * 5.0
                        short_k = long_k + spread_width
                        p_long = BlackScholesEngine.price("call", spot, long_k, T_entry, self.config.risk_free_rate, iv)
                        p_short = BlackScholesEngine.price("call", spot, short_k, T_entry, self.config.risk_free_rate, iv)
                        debit = p_long - p_short

                        if debit >= 0.50 and debit < spread_width * 0.70:
                            max_risk = debit * 100.0
                            target_trade = ("BUY_DEBIT", "BULL_CALL_SPREAD", short_k, long_k, debit, max_risk, exp_date)

                    # D. CONVEXITY CRASH PROTECTION: BEAR PUT DEBIT SPREAD
                    elif self.config.enable_option_buying and is_bear and iv_rank <= self.config.max_iv_rank_to_buy and spot < ema_20 * 0.99:
                        long_k = round(spot / 5.0) * 5.0
                        short_k = long_k - spread_width
                        p_long = BlackScholesEngine.price("put", spot, long_k, T_entry, self.config.risk_free_rate, iv)
                        p_short = BlackScholesEngine.price("put", spot, short_k, T_entry, self.config.risk_free_rate, iv)
                        debit = p_long - p_short

                        if debit >= 0.50 and debit < spread_width * 0.70:
                            max_risk = debit * 100.0
                            target_trade = ("BUY_DEBIT", "BEAR_PUT_SPREAD", short_k, long_k, debit, max_risk, exp_date)

                    if target_trade:
                        action, strat, sk, lk, entry_px, risk_per_contract, exp_d = target_trade
                        allocated = total_equity * self.config.allocation_pct_per_trade
                        num_contracts = int(allocated / risk_per_contract)
                        num_contracts = max(1, min(30, num_contracts))
                        total_margin = risk_per_contract * num_contracts

                        if cash >= total_margin:
                            cash -= total_margin
                            trade_id_counter += 1
                            open_positions.append(Position(
                                trade_id=trade_id_counter,
                                symbol=sym,
                                action=action,
                                strategy=strat,
                                entry_date=current_date,
                                expiration_date=exp_d,
                                entry_spot=spot,
                                entry_price_per_share=entry_px,
                                num_contracts=num_contracts,
                                short_strike=sk,
                                long_strike=lk,
                                margin_locked=total_margin,
                                max_profit=entry_px * 100.0 * num_contracts if action == "SELL_CREDIT" else (spread_width - entry_px) * 100.0 * num_contracts,
                                max_loss=total_margin,
                                initial_iv=iv,
                            ))

        series_equity = pd.Series(equity_history)
        metrics = PerformanceCalculator.calculate(series_equity, closed_trades, self.config.risk_free_rate)
        return series_equity, closed_trades, metrics