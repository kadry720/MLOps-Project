PYTHON ?= python
VALIDATION_CONFIG ?= configs/validation.yaml
DATA_VALIDATION_INPUT ?=

.PHONY: install lint format test validate validate-strict baseline ci

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	ruff check src/validation src/utils src/pipeline tests/unit/test_data_validation_pipeline.py
	black --check src/validation src/utils src/pipeline tests/unit/test_data_validation_pipeline.py
	flake8 src/validation src/utils src/pipeline tests/unit/test_data_validation_pipeline.py

format:
	black src/validation src/utils src/pipeline tests/unit/test_data_validation_pipeline.py
	ruff check src/validation src/utils src/pipeline tests/unit/test_data_validation_pipeline.py --fix

test:
	pytest tests/unit --cov=src --cov-report=term-missing

validate:
	$(PYTHON) src/pipeline/validate_pipeline.py --config $(VALIDATION_CONFIG) $(if $(DATA_VALIDATION_INPUT),--input-data $(DATA_VALIDATION_INPUT),)

validate-strict:
	$(PYTHON) src/pipeline/validate_pipeline.py --config $(VALIDATION_CONFIG) --fail-on-warnings $(if $(DATA_VALIDATION_INPUT),--input-data $(DATA_VALIDATION_INPUT),)

baseline:
	$(PYTHON) src/pipeline/validate_pipeline.py --config $(VALIDATION_CONFIG) --update-baseline

ci: lint test validate
