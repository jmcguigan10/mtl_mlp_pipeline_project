#!/bin/bash
#SBATCH --job-name=mtl-box3d-prep
#SBATCH --account=isaac-utk0307
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=condo-slagergr
#SBATCH --qos=condo
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/nfs/home/jmcguig1/git/mtl_mlp_pipeline_project
#SBATCH -o slurm/output/preprocess-box3d-%j.out
#SBATCH -e slurm/output/preprocess-box3d-%j.err

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load gcc/10.2.0
module load openmpi/4.0.3-gcc
module load hdf5/1.10.7-gcc
module load Python/3.9.10-gcc
module load cuda/11.4.2-gcc

source /nfs/home/jmcguig1/git/mtl_mlp_pipeline_project/.venv/bin/activate

mkdir -p slurm/output

export MTL_MLP_INPUT_DIR=${MTL_MLP_INPUT_DIR:-./example_data}
export MTL_MLP_OUTPUT_DIR=${MTL_MLP_OUTPUT_DIR:-./example_data_box3d_abs}
export MTL_MLP_BATCH_SIZE=${MTL_MLP_BATCH_SIZE:-1024}
export MTL_MLP_MAX_SAMPLES_PER_FILE=${MTL_MLP_MAX_SAMPLES_PER_FILE:-}
export MTL_MLP_STABILITY_THRESHOLD=${MTL_MLP_STABILITY_THRESHOLD:-0.0}
export MTL_MLP_COMPRESSION=${MTL_MLP_COMPRESSION:-lzf}
export MTL_MLP_INCLUDE_BOX3D_FILES=${MTL_MLP_INCLUDE_BOX3D_FILES:-0}
export MTL_MLP_INCLUDE_LEAKAGERATES_FILES=${MTL_MLP_INCLUDE_LEAKAGERATES_FILES:-0}
export MTL_MLP_DISABLE_AMBIGUITY_FILTER=${MTL_MLP_DISABLE_AMBIGUITY_FILTER:-0}
export MTL_MLP_AMBIGUITY_QUANTILE=${MTL_MLP_AMBIGUITY_QUANTILE:-0.02}
export MTL_MLP_AMBIGUITY_DISTANCE_THRESHOLD=${MTL_MLP_AMBIGUITY_DISTANCE_THRESHOLD:-}
export MTL_MLP_AMBIGUITY_STABLE_WEIGHT=${MTL_MLP_AMBIGUITY_STABLE_WEIGHT:-0.0}
export MTL_MLP_AMBIGUITY_ONLY_BC=${MTL_MLP_AMBIGUITY_ONLY_BC:-0}
export MTL_MLP_AMBIGUITY_MAX_UNSTABLE_POINTS=${MTL_MLP_AMBIGUITY_MAX_UNSTABLE_POINTS:-300000}
export MTL_MLP_AMBIGUITY_RANDOM_SEED=${MTL_MLP_AMBIGUITY_RANDOM_SEED:-42}
export MTL_MLP_AMBIGUITY_BRUTEFORCE_CHUNK_SIZE=${MTL_MLP_AMBIGUITY_BRUTEFORCE_CHUNK_SIZE:-4096}

echo "[preprocess] input=${MTL_MLP_INPUT_DIR}"
echo "[preprocess] output=${MTL_MLP_OUTPUT_DIR}"
echo "[preprocess] batch_size=${MTL_MLP_BATCH_SIZE}"
echo "[preprocess] max_samples_per_file=${MTL_MLP_MAX_SAMPLES_PER_FILE:-all}"
echo "[preprocess] stability_threshold=${MTL_MLP_STABILITY_THRESHOLD}"
echo "[preprocess] compression=${MTL_MLP_COMPRESSION}"
echo "[preprocess] include_box3d_files=${MTL_MLP_INCLUDE_BOX3D_FILES}"
echo "[preprocess] include_leakagerates_files=${MTL_MLP_INCLUDE_LEAKAGERATES_FILES}"
echo "[preprocess] disable_ambiguity_filter=${MTL_MLP_DISABLE_AMBIGUITY_FILTER}"
echo "[preprocess] ambiguity_quantile=${MTL_MLP_AMBIGUITY_QUANTILE}"
echo "[preprocess] ambiguity_distance_threshold=${MTL_MLP_AMBIGUITY_DISTANCE_THRESHOLD:-none}"
echo "[preprocess] ambiguity_stable_weight=${MTL_MLP_AMBIGUITY_STABLE_WEIGHT}"
echo "[preprocess] ambiguity_only_bc=${MTL_MLP_AMBIGUITY_ONLY_BC}"
echo "[preprocess] ambiguity_max_unstable_points=${MTL_MLP_AMBIGUITY_MAX_UNSTABLE_POINTS}"
echo "[preprocess] ambiguity_random_seed=${MTL_MLP_AMBIGUITY_RANDOM_SEED}"
echo "[preprocess] ambiguity_bruteforce_chunk_size=${MTL_MLP_AMBIGUITY_BRUTEFORCE_CHUNK_SIZE}"

srun make preprocess \
  INPUT_DIR="${MTL_MLP_INPUT_DIR}" \
  OUTPUT_DIR="${MTL_MLP_OUTPUT_DIR}" \
  BATCH_SIZE="${MTL_MLP_BATCH_SIZE}" \
  MAX_SAMPLES_PER_FILE="${MTL_MLP_MAX_SAMPLES_PER_FILE}" \
  STABILITY_THRESHOLD="${MTL_MLP_STABILITY_THRESHOLD}" \
  COMPRESSION="${MTL_MLP_COMPRESSION}" \
  INCLUDE_BOX3D_FILES="${MTL_MLP_INCLUDE_BOX3D_FILES}" \
  INCLUDE_LEAKAGERATES_FILES="${MTL_MLP_INCLUDE_LEAKAGERATES_FILES}" \
  DISABLE_AMBIGUITY_FILTER="${MTL_MLP_DISABLE_AMBIGUITY_FILTER}" \
  AMBIGUITY_QUANTILE="${MTL_MLP_AMBIGUITY_QUANTILE}" \
  AMBIGUITY_DISTANCE_THRESHOLD="${MTL_MLP_AMBIGUITY_DISTANCE_THRESHOLD}" \
  AMBIGUITY_STABLE_WEIGHT="${MTL_MLP_AMBIGUITY_STABLE_WEIGHT}" \
  AMBIGUITY_ONLY_BC="${MTL_MLP_AMBIGUITY_ONLY_BC}" \
  AMBIGUITY_MAX_UNSTABLE_POINTS="${MTL_MLP_AMBIGUITY_MAX_UNSTABLE_POINTS}" \
  AMBIGUITY_RANDOM_SEED="${MTL_MLP_AMBIGUITY_RANDOM_SEED}" \
  AMBIGUITY_BRUTEFORCE_CHUNK_SIZE="${MTL_MLP_AMBIGUITY_BRUTEFORCE_CHUNK_SIZE}"
