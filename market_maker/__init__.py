"""Market maker package: multi-Greek Avellaneda-Stoikov-style quoting engine.

Public API (see ARCHITECTURE.md for the full contract):

    from market_maker import Book, OptionContract
    from market_maker import RiskAversion, Quote, PricerFn
    from market_maker import generate_quotes, reservation_price, quote_spread
    from market_maker import compute_book_greeks, price_book, price_contract
"""
from market_maker.book import Book, OptionContract
from market_maker.quoting import (
    PricerFn,
    Quote,
    RiskAversion,
    compute_book_greeks,
    generate_quotes,
    price_book,
    price_contract,
    quote_spread,
    reservation_price,
)

__all__ = [
    "Book",
    "OptionContract",
    "PricerFn",
    "Quote",
    "RiskAversion",
    "compute_book_greeks",
    "generate_quotes",
    "price_book",
    "price_contract",
    "quote_spread",
    "reservation_price",
]
