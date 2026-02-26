PYTHON ?= python

.PHONY: test msd_build msd_index sifter_profile

test:
	$(PYTHON) -m pytest -q

msd_build:
	$(PYTHON) ml/train.py

msd_index:
	$(PYTHON) ml/eval.py

sifter_profile:
	$(PYTHON) -m cProfile -s cumtime ml/eval.py
