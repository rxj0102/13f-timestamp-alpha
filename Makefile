# make test        run the tests
# make check       README matches results/ (offline, used in CI)
# make study       rerun the measurements from the committed history and rewrite the README
# make reproduce   rebuild everything from EDGAR; needs EDGAR_USER_AGENT="Your Name your.address@example.com"
.PHONY: setup test check study reproduce

setup:
	uv sync

test:
	uv run python -m pytest -q

check:
	uv run python scripts/filing_timing_study.py --check

study:
	uv run python scripts/filing_timing_study.py

reproduce:
	uv run python scripts/build_manager_universe.py
	uv run python scripts/build_ctr_log.py
	uv run python scripts/filing_timing_study.py
