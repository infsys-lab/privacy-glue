SHELL := /bin/bash

.PHONY: test integration_test

test:
	pytest --cov-report html --cov-report term \
		--html=tests/reports/unit_tests.html --self-contained-html

integration_test:
	N_GPU="$$(($$(printf "%s" "$$CUDA_VISIBLE_DEVICES" | \
				sed 's/^,\+//g; s/,\+$$//g; s/,\+ */,/g' | \
				tr -cd , | wc -c) + 1))" ;\
	if [ -n "$$CUDA_VISIBLE_DEVICES" ] && [ "$$N_GPU" -eq "1" ]; then \
			SUFFIX="single_gpu"; \
	elif [ -n "$$CUDA_VISIBLE_DEVICES" ] && [ "$$N_GPU" -gt "1" ]; then \
			SUFFIX="multi_gpu_$${N_GPU}"; \
	else \
			SUFFIX="cpu"; \
	fi ;\
	pytest -m slow --no-cov \
		--html="tests/reports/integration_tests_$${SUFFIX}.html" \
		--self-contained-html
