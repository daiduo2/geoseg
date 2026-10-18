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

.PHONY: help test check lint map segment batch export edit bench

help:
	@echo "geoseg tasks:"
	@echo "  make test                       run the test suite"
	@echo "  make check                      compile + test (CI gate)"
	@echo "  make lint                       ruff check (advisory; legacy violations remain)"
	@echo "  make map                        print public API and engine registry"
	@echo "  make segment IMAGE=x.png [N=5] [OUT=runs/cli] [NO_VLM=1] [FORCE=1]"
	@echo "  make batch IMAGES=dir [OUT=runs/batch] [N=5]"
	@echo "  make export SESSION=session.json [OUT=runs/export]"
	@echo "  make edit LABELS=labels.npz [IMAGE=x.png] [EDIT_OUT=runs/edit]"
	@echo "  make bench [OUT=runs/benchmark/results.json]"

test:
	$(UV) run pytest -q

check:
	$(UV) run python -m compileall -q src
	$(UV) run pytest -q

lint:
	$(UV) run ruff check src tests scripts

map:
	$(PY) scripts/dev/dev_cli.py map

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
