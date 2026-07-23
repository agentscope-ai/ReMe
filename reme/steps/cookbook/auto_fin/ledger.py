"""Deterministic normalized-NAV portfolio ledger for Auto Fin."""

from __future__ import annotations

import hashlib
import math
from datetime import date, datetime
from typing import Iterable

from ....schema import (
    ActionStatus,
    ActionType,
    InstrumentType,
    PortfolioSnapshot,
    Position,
    PositionMark,
    ProposedAction,
    RunStatus,
    Settlement,
)


def next_trade_date(trade_date: date, open_dates: Iterable[date]) -> date:
    """Return the first known open date after ``trade_date``."""
    if following := sorted(value for value in open_dates if value > trade_date):
        return following[0]
    raise ValueError(f"no next trading date is available after {trade_date.isoformat()}")


class AutoFinLedger:
    """Apply market-return intervals and simulated actions without real prices."""

    def __init__(self, snapshot: PortfolioSnapshot | None = None, *, max_positions: int = 10, slot_weight: float = 0.1):
        if not 1 <= max_positions <= 10:
            raise ValueError("max_positions must be between 1 and 10")
        if not 0.0 < slot_weight <= 1.0:
            raise ValueError("slot_weight must be in (0, 1]")
        self.max_positions = max_positions
        self.slot_weight = slot_weight
        self.snapshot = (snapshot or PortfolioSnapshot()).model_copy(deep=True)

    @staticmethod
    def _clean(value: float) -> float:
        """Keep persisted floating-point values stable across reruns."""
        return round(value, 12)

    def _recalculate_nav(self) -> None:
        self.snapshot.nav = self._clean(
            self.snapshot.cash_nav + sum(position.normalized_value for position in self.snapshot.positions),
        )
        PortfolioSnapshot.model_validate(self.snapshot.model_dump())

    def apply_marks(self, marks: Iterable[PositionMark]) -> float:
        """Apply each unseen interval once and return the portfolio interval return."""
        before = self.snapshot.nav
        by_code: dict[str, list[PositionMark]] = {}
        for mark in marks:
            by_code.setdefault(mark.code, []).append(mark)

        for position in self.snapshot.positions:
            interval_factor = 1.0
            applied = set(position.applied_interval_ids)
            for mark in sorted(by_code.get(position.code, []), key=lambda value: value.interval_end):
                if mark.interval_id in applied:
                    continue
                interval_factor *= 1.0 + mark.interval_return
                position.applied_interval_ids.append(mark.interval_id)
                applied.add(mark.interval_id)
            position.interval_return = self._clean(interval_factor - 1.0)
            position.cumulative_return_factor = self._clean(position.cumulative_return_factor * interval_factor)
            position.cumulative_return = self._clean(position.cumulative_return_factor - 1.0)
            position.normalized_value = self._clean(position.entry_notional * position.cumulative_return_factor)
            position.portfolio_contribution = self._clean(position.normalized_value - position.entry_notional)
            code_marks = by_code.get(position.code, [])
            if code_marks:
                position.marked_at = max(mark.interval_end for mark in code_marks)

        self._recalculate_nav()
        return self._clean(self.snapshot.nav / before - 1.0) if before else 0.0

    def settle(
        self,
        actions: Iterable[ProposedAction],
        *,
        trade_date: date,
        eligible_sell_date: date,
        fill_at: datetime,
        fill_basis: str,
        market_data_complete: bool,
    ) -> list[Settlement]:
        """Settle prior proposals in SELL-before-BUY order."""
        priority = {ActionType.SELL: 0, ActionType.BUY: 1, ActionType.HOLD: 2}
        ordered = sorted(actions, key=lambda action: priority[action.action])
        results: list[Settlement] = []
        target_notional = self._clean(self.snapshot.nav * self.slot_weight)

        for action in ordered:
            if action.action_id in self.snapshot.settled_action_ids:
                continue
            if action.status is not ActionStatus.PROPOSED:
                results.append(
                    Settlement(
                        action_id=action.action_id,
                        action=action.action,
                        code=action.code,
                        status=action.status,
                        fill_basis=fill_basis,
                        reason=action.rejection_reason,
                    ),
                )
                continue
            if not market_data_complete:
                results.append(
                    Settlement(
                        action_id=action.action_id,
                        action=action.action,
                        code=action.code,
                        status=ActionStatus.PENDING_DATA,
                        fill_basis=fill_basis,
                        reason="required market return data is incomplete",
                    ),
                )
                continue

            if action.action is ActionType.SELL:
                result = self._sell(action, trade_date, fill_at, fill_basis)
            elif action.action is ActionType.BUY:
                result = self._buy(action, trade_date, eligible_sell_date, target_notional, fill_at, fill_basis)
            else:
                result = Settlement(
                    action_id=action.action_id,
                    action=action.action,
                    code=action.code,
                    status=ActionStatus.FILLED,
                    fill_basis=fill_basis,
                    filled_at=fill_at,
                    reason="HOLD requires no portfolio mutation",
                )
            results.append(result)
            if result.status is ActionStatus.FILLED:
                self.snapshot.settled_action_ids.append(action.action_id)

        self._recalculate_nav()
        return results

    def _sell(
        self,
        action: ProposedAction,
        trade_date: date,
        fill_at: datetime,
        fill_basis: str,
    ) -> Settlement:
        position = next((item for item in self.snapshot.positions if item.code == action.code), None)
        reason = ""
        if position is None:
            reason = "position is not held"
        elif trade_date < position.eligible_sell_date:
            reason = f"T+1: eligible on {position.eligible_sell_date.isoformat()}"
        if reason:
            return Settlement(
                action_id=action.action_id,
                action=action.action,
                code=action.code,
                status=ActionStatus.REJECTED,
                fill_basis=fill_basis,
                reason=reason,
            )

        assert position is not None
        self.snapshot.cash_nav = self._clean(self.snapshot.cash_nav + position.normalized_value)
        self.snapshot.realized_return = self._clean(
            self.snapshot.realized_return + position.normalized_value - position.entry_notional,
        )
        self.snapshot.positions.remove(position)
        return Settlement(
            action_id=action.action_id,
            action=action.action,
            code=action.code,
            status=ActionStatus.FILLED,
            fill_basis=fill_basis,
            filled_at=fill_at,
        )

    def _buy(
        self,
        action: ProposedAction,
        trade_date: date,
        eligible_sell_date: date,
        target_notional: float,
        fill_at: datetime,
        fill_basis: str,
    ) -> Settlement:
        held_codes = {position.code for position in self.snapshot.positions}
        reason = ""
        if action.code in held_codes:
            reason = "duplicate position"
        elif len(self.snapshot.positions) >= self.max_positions:
            reason = f"position limit {self.max_positions} reached"
        elif self.snapshot.cash_nav + 1e-12 < target_notional:
            reason = "insufficient normalized cash"
        if reason:
            return Settlement(
                action_id=action.action_id,
                action=action.action,
                code=action.code,
                status=ActionStatus.REJECTED,
                fill_basis=fill_basis,
                reason=reason,
            )

        self.snapshot.cash_nav = self._clean(self.snapshot.cash_nav - target_notional)
        self.snapshot.positions.append(
            Position(
                code=action.code,
                name=action.name or action.code,
                instrument_type=action.instrument_type,
                buy_trade_date=trade_date,
                eligible_sell_date=eligible_sell_date,
                entry_notional=target_notional,
                normalized_value=target_notional,
                marked_at=fill_at,
            ),
        )
        return Settlement(
            action_id=action.action_id,
            action=action.action,
            code=action.code,
            status=ActionStatus.FILLED,
            fill_basis=fill_basis,
            filled_at=fill_at,
        )

    def validate_proposals(
        self,
        actions: Iterable[ProposedAction],
        *,
        run_id: str,
        trade_date: date,
        proposed_at: datetime,
        scheduled_fill_at: datetime,
        run_status: RunStatus,
        research_only: bool,
        position_data_complete: bool = True,
    ) -> tuple[list[ProposedAction], list[ProposedAction]]:
        """Enforce phase-one portfolio rules on current analysis proposals."""
        accepted: list[ProposedAction] = []
        rejected: list[ProposedAction] = []
        held_codes = {position.code for position in self.snapshot.positions}
        pending_buy_codes: set[str] = set()
        pending_sell_codes: set[str] = set()
        available_slots = self.max_positions - len(self.snapshot.positions)
        affordable_slots = math.floor((self.snapshot.cash_nav + 1e-12) / (self.snapshot.nav * self.slot_weight))

        for index, original in enumerate(actions):
            action = original.model_copy(deep=True)
            digest = hashlib.sha256(f"{run_id}:{index}:{action.action}:{action.code}".encode()).hexdigest()[:16]
            action.action_id = f"{run_id}:{digest}"
            action.proposed_at = proposed_at
            action.scheduled_fill_at = scheduled_fill_at
            action.status = ActionStatus.PROPOSED
            action.rejection_reason = ""

            reason = ""
            if action.action in {ActionType.BUY, ActionType.SELL} and proposed_at > scheduled_fill_at:
                action.status = ActionStatus.MISSED_CUTOFF
                reason = "analysis completed after the scheduled simulated fill"
            elif research_only and action.action in {ActionType.BUY, ActionType.SELL}:
                reason = "research_only run cannot change simulated holdings"
            elif run_status is not RunStatus.COMPLETE and action.action is ActionType.BUY:
                reason = "degraded or failed analysis cannot add risk"
            elif action.action is ActionType.SELL and not position_data_complete:
                reason = "incomplete position data cannot support a simulated sell"
            elif action.action is ActionType.BUY:
                if action.instrument_type not in {
                    InstrumentType.STOCK,
                    InstrumentType.DOMESTIC_EQUITY_ETF,
                }:
                    reason = "instrument is outside the phase-one universe"
                elif action.code in held_codes or action.code in pending_buy_codes:
                    reason = "duplicate position"
                elif len(pending_buy_codes) >= available_slots:
                    reason = f"position limit {self.max_positions} reached"
                elif len(pending_buy_codes) >= affordable_slots:
                    reason = "insufficient normalized cash"
                else:
                    pending_buy_codes.add(action.code)
            elif action.action is ActionType.SELL:
                position = next((item for item in self.snapshot.positions if item.code == action.code), None)
                if position is None:
                    reason = "position is not held"
                elif trade_date < position.eligible_sell_date:
                    reason = f"T+1: eligible on {position.eligible_sell_date.isoformat()}"
                elif action.code in pending_sell_codes:
                    reason = "duplicate sell proposal"
                else:
                    pending_sell_codes.add(action.code)

            if reason:
                if action.status is ActionStatus.PROPOSED:
                    action.status = ActionStatus.REJECTED
                action.rejection_reason = reason
                rejected.append(action)
            elif action.action is ActionType.HOLD:
                action.status = ActionStatus.FILLED
                accepted.append(action)
            else:
                accepted.append(action)

        return accepted, rejected
