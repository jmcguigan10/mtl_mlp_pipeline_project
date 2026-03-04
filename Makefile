SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python

CONFIG ?= configs/rhea_box3d_abs_train.yaml
RESUME ?=
CHECKPOINT ?= outputs/rhea_box3d_abs_train/checkpoints/best.pt
SPLIT ?= test
OUTPUT ?=
EVAL_OUTPUT ?= outputs/rhea_box3d_abs_train/eval_metrics.json
PRED_OUTPUT ?= outputs/rhea_box3d_abs_train/predictions.npz
FILES ?=

INPUT_DIR ?= ./example_data
OUTPUT_DIR ?= ./example_data_box3d_abs
BATCH_SIZE ?= 1024
MAX_SAMPLES_PER_FILE ?=
STABILITY_THRESHOLD ?= 0.0
COMPRESSION ?= lzf
INCLUDE_BOX3D_FILES ?= 0
INCLUDE_LEAKAGERATES_FILES ?= 0

DISABLE_AMBIGUITY_FILTER ?= 0
AMBIGUITY_QUANTILE ?= 0.02
AMBIGUITY_DISTANCE_THRESHOLD ?=
AMBIGUITY_STABLE_WEIGHT ?= 0.0
AMBIGUITY_ONLY_BC ?= 0
AMBIGUITY_MAX_UNSTABLE_POINTS ?= 300000
AMBIGUITY_RANDOM_SEED ?= 42
AMBIGUITY_BRUTEFORCE_CHUNK_SIZE ?= 4096

SWEEP_CONFIG ?= configs/rhea_v100_sweep.yaml
SWEEP_JSON ?= outputs/tuning/v100_sweep_results.json
SWEEP_CSV ?= outputs/tuning/v100_sweep_results.csv
SWEEP_BATCH_SIZES ?= 256,512,1024,2048,4096,8192
SWEEP_LRS ?= 3e-4,1e-3,3e-3
SWEEP_MODEL_SIZES ?= small,medium,large
SWEEP_BATCH_NORMS ?= 0,1
SWEEP_BATCH_TRAIN_STEPS ?= 150
SWEEP_BATCH_SAMPLE_BUDGET ?= 65536
SWEEP_BATCH_VAL_STEPS ?= 10
SWEEP_HYPER_TRAIN_STEPS ?= 200
SWEEP_HYPER_SAMPLE_BUDGET ?= 65536
SWEEP_VAL_STEPS ?= 40
SWEEP_BATCH_SELECTION_REL_TOL ?= 0.05
SWEEP_SELECTED_BATCH ?=
PLOT_HISTORY ?= outputs/rhea_box3d_abs_train/history.csv
PLOT_DIR ?= plots
PLOT_EVAL_LABEL ?= test
SLURM_ACCOUNT ?= isaac-utk0307
SLURM_PARTITION ?= condo-slagergr
SLURM_QOS ?= condo
SLURM_TIME ?= 02:00:00
SLURM_GRES ?= gpu:v100s:1

.PHONY: help check-python train eval predict preprocess plot-losses \
	smoke smoke-rhea smoke-box3d smoke-equiv test-equiv tune-v100 clean-smoke

help:
	@echo "Targets:"
	@echo "  make train CONFIG=<yaml> [RESUME=<checkpoint>]"
	@echo "  make eval CONFIG=<yaml> CHECKPOINT=<pt> [SPLIT=test] [EVAL_OUTPUT=<json>|OUTPUT=<json>]"
	@echo "  make predict CONFIG=<yaml> CHECKPOINT=<pt> [SPLIT=test] [PRED_OUTPUT=<npz>|OUTPUT=<npz>] [FILES='a.h5 b.h5']"
	@echo "  make preprocess INPUT_DIR=<dir> OUTPUT_DIR=<dir> [BATCH_SIZE=1024] [MAX_SAMPLES_PER_FILE=N]"
	@echo "  make plot-losses [PLOT_HISTORY=<history.csv>] [PLOT_DIR=plots] [PLOT_EVAL_LABEL=test]"
	@echo "  make tune-v100 [SWEEP_CONFIG=<yaml>] [SWEEP_BATCH_SIZES=256,512,...]"
	@echo "  make smoke | smoke-rhea | smoke-box3d | smoke-equiv | test-equiv"
	@echo ""
	@echo "Recommended start:"
	@echo "  make preprocess INPUT_DIR=./example_data OUTPUT_DIR=./example_data_box3d_abs"
	@echo "  make train CONFIG=configs/rhea_box3d_abs_train.yaml"
	@echo "See config docs:"
	@echo "  configs/README.md"

check-python:
	@if [[ ! -x "$(PYTHON)" ]]; then \
		echo "Missing Python interpreter at $(PYTHON)."; \
		echo "Expected .venv symlink to point to a valid environment."; \
		exit 1; \
	fi

train: check-python
	@set -euo pipefail; \
	cmd=( "$(PYTHON)" -u train.py --config "$(CONFIG)" ); \
	if [[ -n "$(strip $(RESUME))" ]]; then \
		cmd+=( --resume "$(RESUME)" ); \
	fi; \
	"$${cmd[@]}"

eval: check-python
	@set -euo pipefail; \
	out="$(EVAL_OUTPUT)"; \
	if [[ -n "$(strip $(OUTPUT))" ]]; then \
		out="$(OUTPUT)"; \
	fi; \
	"$(PYTHON)" -u evaluate.py \
		--config "$(CONFIG)" \
		--checkpoint "$(CHECKPOINT)" \
		--split "$(SPLIT)" \
		--output "$$out"

predict: check-python
	@set -euo pipefail; \
	out="$(PRED_OUTPUT)"; \
	if [[ -n "$(strip $(OUTPUT))" ]]; then \
		out="$(OUTPUT)"; \
	fi; \
	cmd=( "$(PYTHON)" -u predict.py --config "$(CONFIG)" --checkpoint "$(CHECKPOINT)" --split "$(SPLIT)" --output "$$out" ); \
	if [[ -n "$(strip $(FILES))" ]]; then \
		read -r -a extra_files <<< "$(FILES)"; \
		cmd+=( --files "$${extra_files[@]}" ); \
	fi; \
	"$${cmd[@]}"

preprocess: check-python
	@set -euo pipefail; \
	cmd=( "$(PYTHON)" -u scripts/preprocess_box3d_hdf5.py \
		--input_dir "$(INPUT_DIR)" \
		--output_dir "$(OUTPUT_DIR)" \
		--batch_size "$(BATCH_SIZE)" \
		--stability_threshold "$(STABILITY_THRESHOLD)" \
		--compression "$(COMPRESSION)" \
		--ambiguity_quantile "$(AMBIGUITY_QUANTILE)" \
		--ambiguity_stable_weight "$(AMBIGUITY_STABLE_WEIGHT)" \
		--ambiguity_max_unstable_points "$(AMBIGUITY_MAX_UNSTABLE_POINTS)" \
		--ambiguity_random_seed "$(AMBIGUITY_RANDOM_SEED)" \
		--ambiguity_bruteforce_chunk_size "$(AMBIGUITY_BRUTEFORCE_CHUNK_SIZE)" \
		--overwrite ); \
	if [[ -n "$(strip $(MAX_SAMPLES_PER_FILE))" ]]; then \
		cmd+=( --max_samples_per_file "$(MAX_SAMPLES_PER_FILE)" ); \
	fi; \
	if [[ "$(INCLUDE_BOX3D_FILES)" == "1" ]]; then \
		cmd+=( --include_box3d_files ); \
	fi; \
	if [[ "$(INCLUDE_LEAKAGERATES_FILES)" == "1" ]]; then \
		cmd+=( --include_leakagerates_files ); \
	fi; \
	if [[ "$(DISABLE_AMBIGUITY_FILTER)" == "1" ]]; then \
		cmd+=( --disable_ambiguity_filter ); \
	fi; \
	if [[ -n "$(strip $(AMBIGUITY_DISTANCE_THRESHOLD))" ]]; then \
		cmd+=( --ambiguity_distance_threshold "$(AMBIGUITY_DISTANCE_THRESHOLD)" ); \
	fi; \
	if [[ "$(AMBIGUITY_ONLY_BC)" == "1" ]]; then \
		cmd+=( --ambiguity_only_bc ); \
	fi; \
	"$${cmd[@]}"

plot-losses: check-python
	@$(PYTHON) scripts/plot_rhea_style_losses.py \
		--history "$(PLOT_HISTORY)" \
		--output_dir "$(PLOT_DIR)" \
		--eval_label "$(PLOT_EVAL_LABEL)"

smoke: check-python
	@$(PYTHON) scripts/smoke_test.py

smoke-rhea: check-python
	@$(PYTHON) scripts/smoke_test_rhea.py

smoke-box3d: check-python
	@$(PYTHON) scripts/smoke_test_box3d_pipeline.py

smoke-equiv: check-python
	@$(PYTHON) scripts/smoke_test_equivariant_pipeline.py

test-equiv: check-python
	@$(PYTHON) scripts/test_equivariant_basis.py

tune-v100: check-python
	@set -euo pipefail; \
	cmd=( srun -A "$(SLURM_ACCOUNT)" -p "$(SLURM_PARTITION)" --qos="$(SLURM_QOS)" \
		--gres="$(SLURM_GRES)" --time="$(SLURM_TIME)" \
		"$(PYTHON)" -u scripts/v100_tune_sweep.py \
			--base_config "$(SWEEP_CONFIG)" \
			--output_json "$(SWEEP_JSON)" \
			--output_csv "$(SWEEP_CSV)" \
			--batch_sizes "$(SWEEP_BATCH_SIZES)" \
			--learning_rates "$(SWEEP_LRS)" \
			--model_sizes "$(SWEEP_MODEL_SIZES)" \
			--batch_norm_options "$(SWEEP_BATCH_NORMS)" \
			--batch_train_steps "$(SWEEP_BATCH_TRAIN_STEPS)" \
			--batch_sample_budget "$(SWEEP_BATCH_SAMPLE_BUDGET)" \
			--batch_val_steps "$(SWEEP_BATCH_VAL_STEPS)" \
			--hyper_train_steps "$(SWEEP_HYPER_TRAIN_STEPS)" \
			--hyper_sample_budget "$(SWEEP_HYPER_SAMPLE_BUDGET)" \
			--batch_selection_rel_tol "$(SWEEP_BATCH_SELECTION_REL_TOL)" \
			--val_steps "$(SWEEP_VAL_STEPS)" ); \
	if [[ -n "$(strip $(SWEEP_SELECTED_BATCH))" ]]; then \
		cmd+=( --selected_batch_size "$(SWEEP_SELECTED_BATCH)" ); \
	fi; \
	"$${cmd[@]}"

clean-smoke:
	@rm -rf /tmp/mtl_mlp_smoke_* /tmp/mtl_mlp_box3d_smoke_* /tmp/mtl_mlp_equiv_smoke_*
