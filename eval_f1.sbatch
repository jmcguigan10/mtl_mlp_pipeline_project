#!/bin/bash
#SBATCH --job-name=mtl-mlp-eval
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
#SBATCH -o slurm/output/eval-f1-%j.out
#SBATCH -e slurm/output/eval-f1-%j.err

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
#   sbatch --export=ALL,MTL_MLP_CHECKPOINT=outputs/run/checkpoints/best.pt eval_f1.sbatch
export MTL_MLP_CONFIG=${MTL_MLP_CONFIG:-configs/rhea_box3d_abs_train.yaml}
export MTL_MLP_CHECKPOINT=${MTL_MLP_CHECKPOINT:-outputs/rhea_box3d_abs_train/checkpoints/best.pt}
export MTL_MLP_SPLIT=${MTL_MLP_SPLIT:-test}
export MTL_MLP_EVAL_OUTPUT=${MTL_MLP_EVAL_OUTPUT:-outputs/rhea_box3d_abs_train/slurm_eval_metrics.json}

echo "[eval] config=${MTL_MLP_CONFIG}"
echo "[eval] checkpoint=${MTL_MLP_CHECKPOINT}"
echo "[eval] split=${MTL_MLP_SPLIT}"
echo "[eval] output=${MTL_MLP_EVAL_OUTPUT}"

srun make eval \
  CONFIG="${MTL_MLP_CONFIG}" \
  CHECKPOINT="${MTL_MLP_CHECKPOINT}" \
  SPLIT="${MTL_MLP_SPLIT}" \
  OUTPUT="${MTL_MLP_EVAL_OUTPUT}"
