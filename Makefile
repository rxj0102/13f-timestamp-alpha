# make test        run the tests
# make check       README matches results/ (offline, used in CI)
# make hidden     rerun part 1 (hidden positions) from data/snapshot/ and rewrite its part of the README
# make study       rerun part 2 (filing timing) from the committed history and rewrite its part
# make reproduce   rebuild everything from EDGAR; needs EDGAR_USER_AGENT="Your Name your.address@example.com"
.PHONY: setup test check hidden study reproduce

setup:
	uv sync

test:
	uv run python -m pytest -q

check:
	uv run python scripts/render_hidden.py --check
	uv run python scripts/filing_timing_study.py --check

hidden:
	uv run python scripts/hidden_study.py
	uv run python scripts/render_hidden.py

study:
	uv run python scripts/filing_timing_study.py

reproduce:
	uv run python scripts/build_confidential.py
	uv run python scripts/build_prices.py
	uv run python scripts/prepare_hidden.py
	uv run python scripts/hidden_study.py
	uv run python scripts/render_hidden.py
	uv run python scripts/build_manager_universe.py
	uv run python scripts/build_ctr_log.py
	uv run python scripts/filing_timing_study.py
