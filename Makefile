# make reproduce   rebuild the manager universe from EDGAR and rerun the filing timing study
# make check       verify README Key Findings match results/ (fast, used in CI)
# make test        run the unit tests
# EDGAR requires a User-Agent with your name and email: export EDGAR_USER_AGENT="Name email@example.com"
.PHONY: setup reproduce check test

setup:
	uv sync

reproduce:
	uv run python scripts/build_manager_universe.py
	uv run python scripts/filing_timing_study.py

check:
	uv run python scripts/filing_timing_study.py --check

test:
	MPLBACKEND=Agg uv run pytest -q
