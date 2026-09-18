"""Tests for market_maker.book.Book / OptionContract."""
import pytest

from market_maker.book import Book, OptionContract


def _contract(cid="C1", strike=100.0, maturity=0.5, option_type="call"):
    return OptionContract(contract_id=cid, strike=strike, maturity=maturity, option_type=option_type)


def test_add_contract_registers_and_defaults_inventory_zero():
    book = Book()
    book.add_contract(_contract())
    assert book.contract_ids() == ["C1"]
    assert book.get_inventory("C1") == 0.0
    assert book.is_flat()


def test_add_contract_with_starting_quantity():
    book = Book()
    book.add_contract(_contract(), quantity=5.0)
    assert book.get_inventory("C1") == 5.0
    assert not book.is_flat()


def test_get_inventory_unregistered_contract_defaults_to_zero():
    book = Book()
    assert book.get_inventory("nope") == 0.0


def test_set_inventory_updates_existing_contract():
    book = Book()
    book.add_contract(_contract())
    book.set_inventory("C1", -3.0)
    assert book.get_inventory("C1") == -3.0


def test_set_inventory_unknown_contract_raises():
    book = Book()
    with pytest.raises(KeyError):
        book.set_inventory("missing", 1.0)


def test_adjust_inventory_accumulates():
    book = Book()
    book.add_contract(_contract(), quantity=2.0)
    new_qty = book.adjust_inventory("C1", 3.0)
    assert new_qty == 5.0
    assert book.get_inventory("C1") == 5.0

    new_qty = book.adjust_inventory("C1", -10.0)
    assert new_qty == -5.0


def test_adjust_inventory_unknown_contract_raises():
    book = Book()
    with pytest.raises(KeyError):
        book.adjust_inventory("missing", 1.0)


def test_is_flat_multiple_contracts():
    book = Book()
    book.add_contract(_contract("C1"))
    book.add_contract(_contract("C2", strike=110.0))
    assert book.is_flat()
    book.set_inventory("C2", 1.0)
    assert not book.is_flat()
    book.set_inventory("C2", 0.0)
    assert book.is_flat()
