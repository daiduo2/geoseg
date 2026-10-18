# geoseg developer tasks.
#
# Canonical commands live here so contributors do not have to remember which
# script under scripts/ implements a task. `just` is not installed on this
# machine, so this is a Makefile.

SHELL := /bin/bash
UV ?= uv
PY := $(UV) run python

IMAGE ?=
N ?= 5
OUT ?= runs/cli
NO_VLM ?=
FORCE ?=
IMAGES ?=
SESSION ?=
LABELS ?=
EDIT_OUT ?= runs/edit
RUN ?=
TXT_DIR ?=
DATASETS ?=
PANEL ?=
DRAWIO ?=

.PHONY: help test check lint map tools doctor segment batch export edit bench sandbox export-txt reconstruct e2e docs corpus corpus-check best history review usage

help:
	@echo "geoseg tasks:"
	@echo "  make test                       run the test suite"
	@echo "  make check                      lint + compile + test (CI gate)"
	@echo "  make lint                       ruff check (configured in pyproject.toml)"
	@echo "  make map                        print public API and engine registry"
	@echo "  make tools                      list maintained scripts with their purpose"
	@echo "  make doctor                     environment health check"
	@echo "  make segment IMAGE=x.png [N=5] [OUT=runs/cli] [NO_VLM=1] [FORCE=1]"
	@echo "  make batch IMAGES=dir [OUT=runs/batch] [N=5]"
	@echo "  make export SESSION=session.json [OUT=runs/export]"
	@echo "  make edit LABELS=labels.npz [IMAGE=x.png] [EDIT_OUT=runs/edit]"
	@echo "  make sandbox IMAGE=x.png PANEL=0  single-panel sandbox segmentation"
	@echo "  make export-txt RUN=run_dir [OUT=run_dir/txt_export]"
	@echo "  make reconstruct TXT_DIR=dir [OUT=dir]"
	@echo "  make e2e [DATASETS='a b'] [NO_VLM=1]"
	@echo "  make docs DRAWIO='docs/assets/flowcharts/*.drawio'"
	@echo "  make corpus                     rebuild validation/ from runs/"
	@echo "  make corpus-check               verify corpus files and hashes"
	@echo "  make best                       rebuild validation/best (most-iterated panels)"
	@echo "  make history                    show panel -> campaign/run history"
	@echo "  make usage                      rank panels by export and review evidence"
	@echo "  make review                     regenerate numbered review sheets"
	@echo "  make bench [OUT=runs/benchmark/results.json]"

test:
	$(UV) run pytest -q

check: lint
	$(UV) run python -m compileall -q src
	$(UV) run pytest -q

lint:
	$(UV) run ruff check .

map:
	$(PY) scripts/dev/dev_cli.py map

tools:
	$(PY) scripts/dev/dev_cli.py tools

segment:
	@test -n "$(IMAGE)" || { echo "usage: make segment IMAGE=path/to/panel.png [N=5] [OUT=runs/cli]"; exit 1; }
	$(PY) scripts/dev/dev_cli.py segment "$(IMAGE)" --n-layers $(N) --out "$(OUT)" $(if $(NO_VLM),--no-vlm,) $(if $(FORCE),--force,)

batch:
	@test -n "$(IMAGES)" || { echo "usage: make batch IMAGES=dir [OUT=runs/batch] [N=5]"; exit 1; }
	$(PY) -m geoseg.batch --images_dir "$(IMAGES)" --output_dir "$(OUT)" --n_layers $(N)

export:
	@test -n "$(SESSION)" || { echo "usage: make export SESSION=session.json [OUT=runs/export]"; exit 1; }
	$(PY) -m geoseg.batch --session "$(SESSION)" --export_only --output_dir "$(OUT)"

edit:
	@test -n "$(LABELS)" || { echo "usage: make edit LABELS=labels.npz [IMAGE=x.png]"; exit 1; }
	@mkdir -p "$(EDIT_OUT)"
	$(PY) -m geoseg.modules.editor.napari_app --labels "$(LABELS)" $(if $(IMAGE),--image "$(IMAGE)",) \
		--output-labels "$(EDIT_OUT)/labels_edited.npz" --output-shapes "$(EDIT_OUT)/shapes.json"

bench:
	$(PY) scripts/dev/dev_cli.py bench $(if $(OUT),--out "$(OUT)",)

doctor:
	$(PY) scripts/env_check.py

sandbox:
	@test -n "$(IMAGE)" -a -n "$(PANEL)" || { echo "usage: make sandbox IMAGE=panel.png PANEL=0"; exit 1; }
	$(PY) scripts/parallel_segment.py --image "$(IMAGE)" --panel-id "$(PANEL)"

export-txt:
	@test -n "$(RUN)" || { echo "usage: make export-txt RUN=run_dir [OUT=dir]"; exit 1; }
	$(PY) scripts/export_segmentation_txt.py "$(RUN)" $(if $(OUT),--output-dir "$(OUT)",)

reconstruct:
	@test -n "$(TXT_DIR)" || { echo "usage: make reconstruct TXT_DIR=dir [OUT=dir]"; exit 1; }
	$(PY) scripts/reconstruct_from_txt.py "$(TXT_DIR)" $(if $(OUT),--output-dir "$(OUT)",)

e2e:
	$(PY) scripts/run_e2e_batch.py $(if $(DATASETS),--datasets $(DATASETS),) $(if $(NO_VLM),--no-vlm,)

docs:
	@test -n "$(DRAWIO)" || { echo "usage: make docs DRAWIO='docs/assets/flowcharts/*.drawio'"; exit 1; }
	$(PY) scripts/drawio_to_png.py $(DRAWIO)

corpus:
	$(PY) scripts/dev/build_corpus.py --clean

corpus-check:
	$(PY) scripts/dev/corpus.py check --hashes

best:
	$(PY) scripts/dev/build_best_examples.py

history:
	$(PY) scripts/dev/panel_history.py

review:
	$(PY) scripts/dev/make_review_sheets.py

usage:
	$(PY) scripts/dev/export_view_history.py
