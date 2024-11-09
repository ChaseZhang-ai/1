.PHONY: test figures demo
test:
	python -m pytest
figures:
	python tools/generate_figures.py
demo:
	python tools/run_demo.py
