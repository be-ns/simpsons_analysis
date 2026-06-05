# Reproducible pipeline for the Simpsons IMDb analysis.
# Every target is deterministic given the fixed seeds in src/simpsons.
.PHONY: help install analysis autoresearch beat-chronology eval-llm llm-extract llm-collect model app all clean

PY := PYTHONPATH=src python3

help:
	@echo "make install         install dependencies (requirements.txt)"
	@echo "make analysis        honest metrics + core figures      -> reports/"
	@echo "make autoresearch    42-pipeline model/feature search    -> reports/"
	@echo "make beat-chronology detrended-residual test (the real bar)"
	@echo "make eval-llm        evaluate blind LLM rubric features vs chronology"
	@echo "make llm-extract     submit Opus rubric extraction (Batch API; needs creds)"
	@echo "make llm-collect B=<batch_id>   fetch a finished extraction batch"
	@echo "make model           train + persist the rating model    -> models/"
	@echo "make app             run the recommender demo at :8080"
	@echo "make all             analysis + autoresearch + beat-chronology + model"

install:
	pip install -r requirements.txt

analysis:
	$(PY) scripts/run_analysis.py

autoresearch:
	$(PY) scripts/autoresearch.py $(ITERS)

beat-chronology:
	$(PY) scripts/beat_chronology.py

eval-llm:
	$(PY) scripts/eval_llm.py

llm-extract:
	$(PY) -m simpsons.llm_features extract $(LIMIT)

llm-collect:
	$(PY) -m simpsons.llm_features collect $(B)

model:
	$(PY) scripts/train_model.py

app:
	$(PY) web_app.py

all: analysis autoresearch beat-chronology model

clean:
	rm -rf reports/figures/*.png reports/*.json reports/*.csv
