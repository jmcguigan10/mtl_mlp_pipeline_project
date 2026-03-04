#!/bin/bash
#SBATCH --job-name=mtl-mlp-predict
#SBATCH --account=isaac-utk0307
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=condo-slagergr
#SBATCH --qos=condo
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/nfs/home/jmcguig1/git/mtl_mlp_pipeline_project
#SBATCH -o slurm/output/predict-%j.out
#SBATCH -e slurm/output/predict-%j.err

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load gcc/10.2.0
module load openmpi/4.0.3-gcc
module load hdf5/1.10.7-gcc
module load Python/3.9.10-gcc
module load cuda/11.4.2-gcc

source /nfs/home/jmcguig1/git/mtl_mlp_pipeline_project/.venv/bin/activate

mkdir -p slurm/output

# Override at submission time:
#   sbatch --export=ALL,MTL_MLP_CHECKPOINT=outputs/run/checkpoints/best.pt predict.sbatch
export MTL_MLP_CONFIG=${MTL_MLP_CONFIG:-configs/rhea_box3d_abs_train.yaml}
export MTL_MLP_CHECKPOINT=${MTL_MLP_CHECKPOINT:-outputs/rhea_box3d_abs_train/checkpoints/best.pt}
export MTL_MLP_SPLIT=${MTL_MLP_SPLIT:-test}
export MTL_MLP_PRED_OUTPUT=${MTL_MLP_PRED_OUTPUT:-outputs/rhea_box3d_abs_train/slurm_predictions.npz}
export MTL_MLP_FILES=${MTL_MLP_FILES:-}

echo "[predict] config=${MTL_MLP_CONFIG}"
echo "[predict] checkpoint=${MTL_MLP_CHECKPOINT}"
echo "[predict] split=${MTL_MLP_SPLIT}"
echo "[predict] output=${MTL_MLP_PRED_OUTPUT}"
if [[ -n "${MTL_MLP_FILES}" ]]; then
  echo "[predict] files=${MTL_MLP_FILES}"
fi

srun make predict \
  CONFIG="${MTL_MLP_CONFIG}" \
  CHECKPOINT="${MTL_MLP_CHECKPOINT}" \
  SPLIT="${MTL_MLP_SPLIT}" \
  OUTPUT="${MTL_MLP_PRED_OUTPUT}" \
  FILES="${MTL_MLP_FILES}"
