PYTHON ?= python
VALIDATION_CONFIG ?= configs/validation.yaml
DATA_VALIDATION_INPUT ?=

.PHONY: install lint format test validate validate-model validate-strict baseline ci

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	ruff check src/validation src/utils src/pipeline src/evaluation src/orchestration tests/unit/test_data_validation_pipeline.py tests/unit/test_model_validation.py tests/unit/test_model_service.py tests/unit/test_preprocessing_unit.py
	black --check src/validation src/utils src/pipeline src/evaluation src/orchestration tests/unit/test_data_validation_pipeline.py tests/unit/test_model_validation.py tests/unit/test_model_service.py tests/unit/test_preprocessing_unit.py
	flake8 src/validation src/utils src/pipeline src/evaluation src/orchestration tests/unit/test_data_validation_pipeline.py tests/unit/test_model_validation.py tests/unit/test_model_service.py tests/unit/test_preprocessing_unit.py

format:
	black src/validation src/utils src/pipeline src/evaluation src/orchestration tests/unit/test_data_validation_pipeline.py tests/unit/test_model_validation.py tests/unit/test_model_service.py tests/unit/test_preprocessing_unit.py
	ruff check src/validation src/utils src/pipeline src/evaluation src/orchestration tests/unit/test_data_validation_pipeline.py tests/unit/test_model_validation.py tests/unit/test_model_service.py tests/unit/test_preprocessing_unit.py --fix

test:
	pytest tests/unit --cov=src --cov-report=term-missing --cov-fail-under=70

validate:
	$(PYTHON) src/pipeline/validate_pipeline.py --config $(VALIDATION_CONFIG) $(if $(DATA_VALIDATION_INPUT),--input-data $(DATA_VALIDATION_INPUT),)

validate-model:
	$(PYTHON) src/evaluation/validate_model.py --config configs/params.yaml --validation-config configs/validation.yaml --allow-metrics-fallback

validate-strict:
	$(PYTHON) src/pipeline/validate_pipeline.py --config $(VALIDATION_CONFIG) --fail-on-warnings $(if $(DATA_VALIDATION_INPUT),--input-data $(DATA_VALIDATION_INPUT),)

baseline:
	$(PYTHON) src/pipeline/validate_pipeline.py --config $(VALIDATION_CONFIG) --update-baseline

ci: lint test validate validate-model
