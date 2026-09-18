"""Book/inventory state representation for the market-making engine.

An `OptionContract` describes *what* is quoted (strike/maturity/type); a
`Book` tracks *how much* of each contract the market maker currently holds
(signed inventory: positive = long, negative = short) plus per-contract
model parameters needed to price it.

This module has no dependency on `pricing` or `vol_models` — it is a plain
data container. The pricer is only ever passed in by the caller at quote
time (see `market_maker.quoting`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OptionContract:
    """Static specification of a single quotable option contract.

    Attributes:
        contract_id: caller-chosen unique key (e.g. "SPX_4500C_30D").
        strike: strike price, same units as spot.
        maturity: time to expiry in YEARS (matches `pricing.api.price`'s
            `maturity` convention).
        option_type: "call" or "put".
    """

    contract_id: str
    strike: float
    maturity: float
    option_type: OptionType = "call"


@dataclass
class Book:
    """Mutable market-maker state: which contracts are quoted and how much
    of each is currently held.

    Shape:
        contracts: dict[contract_id -> OptionContract] — the universe of
            contracts actively quoted. A contract must be registered here
            (via `add_contract`) before it can be quoted or before its
            inventory can be set.
        inventory: dict[contract_id -> float] — signed quantity held.
            Positive = long (market maker bought), negative = short (market
            maker sold). Units are option contracts (not underlying shares);
            a contract with quantity 1.0 and delta 0.5 contributes 0.5
            underlying-equivalent units of delta exposure. Any contract
            without an explicit entry is treated as zero inventory.

    Invariants:
        - `inventory` may only contain keys that also exist in `contracts`
          (enforced by `set_inventory` / `adjust_inventory`).
        - Nothing here is thread-safe; the book is meant to be driven by a
          single-threaded quoting/backtest loop.
    """

    contracts: dict[str, OptionContract] = field(default_factory=dict)
    inventory: dict[str, float] = field(default_factory=dict)

    def add_contract(self, contract: OptionContract, quantity: float = 0.0) -> None:
        """Register a contract as quotable and set its starting inventory."""
        self.contracts[contract.contract_id] = contract
        self.inventory[contract.contract_id] = quantity

    def set_inventory(self, contract_id: str, quantity: float) -> None:
        """Overwrite the signed inventory for an already-registered contract."""
        self._require_known(contract_id)
        self.inventory[contract_id] = quantity

    def adjust_inventory(self, contract_id: str, delta_quantity: float) -> float:
        """Add `delta_quantity` (signed) to a contract's inventory; returns the new total."""
        self._require_known(contract_id)
        new_qty = self.inventory.get(contract_id, 0.0) + delta_quantity
        self.inventory[contract_id] = new_qty
        return new_qty

    def get_inventory(self, contract_id: str) -> float:
        """Current signed inventory for a contract (0.0 if never set)."""
        return self.inventory.get(contract_id, 0.0)

    def contract_ids(self) -> list[str]:
        """All registered contract ids, in insertion order."""
        return list(self.contracts.keys())

    def is_flat(self) -> bool:
        """True if every registered contract has exactly zero inventory."""
        return all(qty == 0.0 for qty in self.inventory.values())

    def _require_known(self, contract_id: str) -> None:
        if contract_id not in self.contracts:
            raise KeyError(
                f"contract_id {contract_id!r} is not registered; call add_contract() first"
            )
