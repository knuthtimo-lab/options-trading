"""
Virtual Paper Trading Portfolio & Kelly Position Sizing Calculator
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


DATA_FILE = Path("paper_portfolio.json")


class PositionSizer:
    @staticmethod
    def calculate_sizing(
        account_equity: float,
        risk_mode: str,          # "HALF_KELLY", "FULL_KELLY", "FIXED_1PCT", "FIXED_2PCT", "AGGRESSIVE_25PCT"
        max_loss_per_contract: float,
        max_profit_per_contract: float,
        win_prob_pct: float,
    ) -> Dict[str, Any]:
        """
        Computes exact position size based on Kelly Criterion or Fixed Risk.
        """
        p = win_prob_pct / 100.0
        q = 1.0 - p
        b = max_profit_per_contract / max_loss_per_contract if max_loss_per_contract > 0 else 1.0

        # Kelly fraction: f* = (bp - q) / b
        kelly_fraction = max(0.0, (b * p - q) / b) if b > 0 else 0.0

        if risk_mode == "FULL_KELLY":
            alloc_pct = min(0.35, kelly_fraction)
        elif risk_mode == "HALF_KELLY":
            alloc_pct = min(0.20, kelly_fraction * 0.5)
        elif risk_mode == "FIXED_1PCT":
            alloc_pct = 0.01 / (max_loss_per_contract / account_equity) if max_loss_per_contract > 0 else 0.05
            alloc_pct = min(0.10, alloc_pct)
        elif risk_mode == "FIXED_2PCT":
            alloc_pct = 0.02 / (max_loss_per_contract / account_equity) if max_loss_per_contract > 0 else 0.10
            alloc_pct = min(0.20, alloc_pct)
        else:  # AGGRESSIVE_25PCT (Master Model parameter)
            alloc_pct = 0.25

        max_risk_dollar = account_equity * alloc_pct
        num_contracts = int(max_risk_dollar / max_loss_per_contract) if max_loss_per_contract > 0 else 1
        num_contracts = max(1, min(50, num_contracts))
        total_risk_dollar = num_contracts * max_loss_per_contract

        return {
            "risk_mode": risk_mode,
            "recommended_contracts": num_contracts,
            "total_risk_dollar": round(total_risk_dollar, 2),
            "portfolio_risk_pct": round((total_risk_dollar / account_equity) * 100.0, 2),
            "expected_profit_dollar": round(num_contracts * max_profit_per_contract, 2),
            "kelly_fraction_full": round(kelly_fraction * 100.0, 1),
            "kelly_fraction_half": round(kelly_fraction * 50.0, 1),
        }


class PaperPortfolio:
    @staticmethod
    def _load_data() -> Dict[str, Any]:
        if not DATA_FILE.exists():
            default_data = {
                "initial_cash": 25000.0,
                "current_cash": 25000.0,
                "active_trades": [],
                "history": [],
            }
            DATA_FILE.write_text(json.dumps(default_data, indent=2), encoding="utf-8")
            return default_data
        try:
            return json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"initial_cash": 25000.0, "current_cash": 25000.0, "active_trades": [], "history": []}

    @staticmethod
    def _save_data(data: Dict[str, Any]):
        DATA_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def get_portfolio(cls) -> Dict[str, Any]:
        data = cls._load_data()
        active = data.get("active_trades", [])
        history = data.get("history", [])
        total_margin = sum(t.get("margin_locked", 0.0) for t in active)
        current_cash = data.get("current_cash", 25000.0)
        total_equity = current_cash + total_margin

        # Ensure trade_id / id and strategy_name / strategy aliases
        for t in active:
            if "trade_id" not in t and "id" in t:
                t["trade_id"] = t["id"]
            if "strategy_name" not in t and "strategy" in t:
                t["strategy_name"] = t["strategy"]
            if "target_price" not in t and "target_exit_price" in t:
                t["target_price"] = t["target_exit_price"]
            if "stop_loss" not in t and "stop_loss_price" in t:
                t["stop_loss"] = t["stop_loss_price"]

        for t in history:
            if "trade_id" not in t and "id" in t:
                t["trade_id"] = t["id"]
            if "strategy_name" not in t and "strategy" in t:
                t["strategy_name"] = t["strategy"]

        realized_pnl = sum(t.get("realized_pnl", 0.0) for t in history)
        winning_trades = sum(1 for t in history if t.get("realized_pnl", 0.0) > 0)
        win_rate = (winning_trades / len(history) * 100.0) if history else 0.0

        return {
            "cash": round(current_cash, 2),
            "cash_balance": round(current_cash, 2),
            "margin_locked": round(total_margin, 2),
            "allocated_margin": round(total_margin, 2),
            "total_equity": round(total_equity, 2),
            "realized_pnl": round(realized_pnl, 2),
            "win_rate_pct": round(win_rate, 1),
            "total_closed_trades": len(history),
            "active_count": len(active),
            "closed_count": len(history),
            "active_trades": active,
            "open_positions": active,
            "history": history,
            "closed_trades": history,
        }

    @classmethod
    def add_trade(cls, trade_dict: Dict[str, Any]) -> Dict[str, Any]:
        data = cls._load_data()
        contracts = trade_dict.get("contracts", 1)
        max_loss = trade_dict.get("max_loss", 500.0)
        margin_needed = max_loss * contracts

        trade_id = len(data["active_trades"]) + len(data["history"]) + 1
        new_trade = {
            "id": trade_id,
            "trade_id": trade_id,
            "symbol": trade_dict.get("symbol", "SPY"),
            "strategy": trade_dict.get("strategy_name", "Spread"),
            "strategy_name": trade_dict.get("strategy_name", "Spread"),
            "action": trade_dict.get("action", "SELL (CREDIT)"),
            "legs": trade_dict.get("legs", ""),
            "expiration": trade_dict.get("expiration", ""),
            "dte": trade_dict.get("dte", 30),
            "entry_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "entry_price": trade_dict.get("entry_price", trade_dict.get("entry_limit_price", 1.0)),
            "contracts": contracts,
            "margin_locked": round(margin_needed, 2),
            "target_exit_price": trade_dict.get("target_price", trade_dict.get("target_exit_price", 0.5)),
            "target_price": trade_dict.get("target_price", trade_dict.get("target_exit_price", 0.5)),
            "stop_loss_price": trade_dict.get("stop_loss", trade_dict.get("stop_loss_price", 2.0)),
            "stop_loss": trade_dict.get("stop_loss", trade_dict.get("stop_loss_price", 2.0)),
            "max_profit": round(trade_dict.get("max_profit", 500.0), 2),
            "max_loss": round(margin_needed, 2),
            "confidence_grade": trade_dict.get("confidence_grade", "A"),
            "status": "OPEN",
        }

        data["current_cash"] -= margin_needed
        data["active_trades"].append(new_trade)
        cls._save_data(data)
        return new_trade

    @classmethod
    def close_trade(cls, trade_id: int, exit_price: float, reason: str = "MANUAL_CLOSE") -> Optional[Dict[str, Any]]:
        data = cls._load_data()
        target = None
        remaining = []
        for t in data["active_trades"]:
            if t["id"] == trade_id:
                target = t
            else:
                remaining.append(t)

        if not target:
            return None

        contracts = target["contracts"]
        entry_price = target["entry_price"]
        is_sell = "SELL" in target["action"]

        if is_sell:
            # PnL = (Entry - Exit) * 100 * contracts
            pnl = (entry_price - exit_price) * 100.0 * contracts
        else:
            pnl = (exit_price - entry_price) * 100.0 * contracts

        target["exit_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        target["exit_price"] = exit_price
        target["realized_pnl"] = round(pnl, 2)
        target["close_reason"] = reason
        target["status"] = "CLOSED"

        # Return margin + PnL
        data["current_cash"] += target["margin_locked"] + pnl
        data["active_trades"] = remaining
        data["history"].append(target)
        cls._save_data(data)
        return target