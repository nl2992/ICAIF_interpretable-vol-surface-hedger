# Interpretable Volatility-Surface Hedger — reproducible workflow
#
#   make install     editable install with dev extras
#   make test        run the test suite
#   make run         full experiment (data -> train -> evaluate -> report)
#   make run-fast    full experiment without ablations (quicker)
#   make staged      run the four staged scripts in sequence
#   make reproduce   install + test + run (clean-room reproduction)
#   make clean       remove regenerable artifacts (keeps committed reports)

PY      ?= python3
CONFIG  ?= configs/experiment.yaml
EXP_ID  ?= ivsh_demo

.PHONY: install test run run-fast staged reproduce clean

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

run:
	$(PY) scripts/run_experiment.py --config $(CONFIG)

run-fast:
	$(PY) scripts/run_experiment.py --config $(CONFIG) --no-ablations

staged:
	$(PY) scripts/build_dataset.py --config $(CONFIG)
	$(PY) scripts/train.py         --config $(CONFIG)
	$(PY) scripts/evaluate.py      --config $(CONFIG)
	$(PY) scripts/make_report.py   --experiment_id $(EXP_ID)

reproduce: install test run

clean:
	rm -rf artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
