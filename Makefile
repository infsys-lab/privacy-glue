SHELL := /bin/bash

.PHONY: test integration

test:
	pytest --cov-report html --cov-report term \
	  --html=tests/reports/unit_tests.html --self-contained-html

integration:
	pytest -m slow --no-cov \
	  --html=tests/reports/integration_tests.html \
	  --self-contained-html
