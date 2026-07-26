"""
fees.py — Coinbase fee model and fee-aware trade decision helpers.

Coinbase Advanced Trade uses a volume-tiered fee schedule:
  Tier 1 (<$10K 30d volume): taker 0.60%, maker 0.40%
  Tier 2 ($10K-$50K):         taker 0.40%, maker 0.25%
  Tier 3 ($50K-$100K):        taker 0.25%, maker 0.15%
  Tier 4 ($100K-$1M):         taker 0.15%, maker 0.10%
  Tier 5 (>$1M):              taker 0.12%, maker 0.08%

This module computes the fee impact on every trade decision:
  - Round-trip cost (entry + exit)
  - Effective R:R after fees
  - Minimum edge required to overcome fees
  - Profitability gate: only enter if expected edge > fees × safety multiple
"""
from dataclasses import dataclass


@dataclass
class FeeModel:
    """Coinbase fee tier and calculations."""

    # Default: Tier 1 (retail, <$10K 30d volume)
    taker_fee_pct: float = 0.60
    maker_fee_pct: float = 0.40
    tier_name: str = "Tier 1 (<$10K)"

    @classmethod
    def from_volume(cls, volume_30d_usd: float) -> "FeeModel":
        """Select fee tier based on 30-day trading volume."""
        if volume_30d_usd >= 1_000_000:
            return cls(taker_fee_pct=0.12, maker_fee_pct=0.08, tier_name="Tier 5 (>$1M)")
        elif volume_30d_usd >= 100_000:
            return cls(taker_fee_pct=0.15, maker_fee_pct=0.10, tier_name="Tier 4 ($100K-$1M)")
        elif volume_30d_usd >= 50_000:
            return cls(taker_fee_pct=0.25, maker_fee_pct=0.15, tier_name="Tier 3 ($50K-$100K)")
        elif volume_30d_usd >= 10_000:
            return cls(taker_fee_pct=0.40, maker_fee_pct=0.25, tier_name="Tier 2 ($10K-$50K)")
        else:
            return cls()  # Tier 1 defaults

    def round_trip_cost_pct(self, order_type: str = "taker") -> float:
        """Total fee cost for entry + exit, as percentage of position value."""
        fee = self.taker_fee_pct if order_type == "taker" else self.maker_fee_pct
        return fee * 2  # entry + exit

    def fee_per_side_pct(self, order_type: str = "taker") -> float:
        """Fee for a single order (entry or exit)."""
        return self.taker_fee_pct if order_type == "taker" else self.maker_fee_pct

    def effective_rr(
        self,
        stop_loss_pct: float,
        take_profit_pct: float,
        order_type: str = "taker",
    ) -> dict:
        """
        Compute effective R:R after fees.

        Args:
            stop_loss_pct: stop loss distance as % (e.g. 3.0 = 3%)
            take_profit_pct: take profit distance as % (e.g. 6.0 = 6%)
            order_type: "taker" (market) or "maker" (limit)

        Returns:
            {
                "gross_rr": float,         # R:R before fees
                "net_rr": float,           # R:R after fees
                "net_target_pct": float,   # profit % after fees
                "net_stop_pct": float,     # loss % after fees
                "round_trip_cost_pct": float,  # total fee cost
                "viable": bool,            # net_rr >= 1.0 (still profitable after fees)
            }
        """
        rtc = self.round_trip_cost_pct(order_type)

        # On entry: fee is paid on position value
        # On exit: fee is paid on exit value (slightly different but we approximate)
        # Net profit = take_profit - entry_fee - exit_fee
        # Net loss = stop_loss + entry_fee + exit_fee

        net_target = take_profit_pct - rtc
        net_stop = stop_loss_pct + rtc

        gross_rr = take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 0
        net_rr = net_target / net_stop if net_stop > 0 else 0

        return {
            "gross_rr": round(gross_rr, 3),
            "net_rr": round(net_rr, 3),
            "net_target_pct": round(net_target, 3),
            "net_stop_pct": round(net_stop, 3),
            "round_trip_cost_pct": round(rtc, 3),
            "viable": net_rr >= 1.0,
        }

    def edge_required(self, safety_multiple: float = 2.0) -> float:
        """
        Minimum expected edge (in %) required to justify a trade.

        The trade must have an expected price move greater than the round-trip
        fee cost multiplied by a safety multiple (default 2x).

        Args:
            safety_multiple: how many times the fee cost the edge must exceed

        Returns:
            minimum edge in percentage points (e.g. 2.4 = need 2.4% expected move)
        """
        return self.round_trip_cost_pct("taker") * safety_multiple

    def should_trade(
        self,
        signal_strength: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        safety_multiple: float = 2.0,
    ) -> dict:
        """
        Fee-aware trade decision gate.

        Args:
            signal_strength: the composite signal score (0-1 scale)
            stop_loss_pct: stop distance in %
            take_profit_pct: target distance in %
            safety_multiple: minimum edge = fees × this multiple

        Returns:
            {
                "approved": bool,
                "reason": str,
                "edge_required": float,
                "estimated_edge": float,
                "rr_after_fees": dict,
            }
        """
        rr = self.effective_rr(stop_loss_pct, take_profit_pct)

        # Estimated edge: signal strength × target distance
        # (stronger signal = higher probability of hitting target)
        estimated_edge = signal_strength * take_profit_pct

        # Required edge: round-trip fees × safety multiple
        required = self.edge_required(safety_multiple)

        approved = False
        reason = ""

        if not rr["viable"]:
            reason = f"REJECT: R:R after fees = {rr['net_rr']:.2f} < 1.0 (fees eat the edge)"
        elif estimated_edge < required:
            reason = (f"REJECT: estimated edge {estimated_edge:.2f}% < required {required:.2f}% "
                      f"(fees×{safety_multiple}={required:.2f}%)")
        else:
            approved = True
            reason = (f"APPROVED: edge {estimated_edge:.2f}% >= {required:.2f}% required, "
                      f"net R:R = {rr['net_rr']:.2f}")

        return {
            "approved": approved,
            "reason": reason,
            "edge_required": round(required, 3),
            "estimated_edge": round(estimated_edge, 3),
            "rr_after_fees": rr,
        }

    def compute_fee_cost(self, position_value: float, order_type: str = "taker") -> float:
        """Compute dollar fee cost for a single order."""
        fee_pct = self.fee_per_side_pct(order_type)
        return position_value * fee_pct / 100

    def summary(self) -> str:
        return (f"Coinbase fees {self.tier_name}: "
                f"taker={self.taker_fee_pct}%, maker={self.maker_fee_pct}%, "
                f"round-trip={self.round_trip_cost_pct()}%")
