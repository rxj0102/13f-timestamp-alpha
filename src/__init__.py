"""
src — 13F Timestamp Arbitrage strategy package.

Modules
-------
edgar_scraper   : EDGAR API client; fetches filing history and holdings XML.
filing_parser   : Parses 13F XML; resolves CUSIPs; computes position deltas.
features        : Conviction score construction; cross-manager aggregation.
signal          : Signal construction, staleness decay, hypothesis testing.
backtest        : Long/short backtesting engine with stratified analysis.
visualize       : Publication-quality charts for the strategy paper.
utils           : Rate limiter, look-ahead lag guard, date utilities.
"""

__version__ = "0.1.0"
__author__  = "13F Timestamp Alpha Research"
