#!/bin/bash
#SBATCH --job-name=mtl-mlp-train
#SBATCH --account=isaac-utk0307
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=condo-slagergr
#SBATCH --qos=condo
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --chdir=/nfs/home/jmcguig1/git/mtl_mlp_pipeline_project
#SBATCH -o slurm/output/train-%j.out
#SBATCH -e slurm/output/train-%j.err

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
#   sbatch --export=ALL,MTL_MLP_CONFIG=configs/rhea_box3d_abs_train.yaml train.sbatch
export MTL_MLP_CONFIG=${MTL_MLP_CONFIG:-configs/rhea_box3d_abs_train.yaml}
export MTL_MLP_RESUME=${MTL_MLP_RESUME:-}

CMD=(srun make train CONFIG="${MTL_MLP_CONFIG}")
if [[ -n "${MTL_MLP_RESUME}" ]]; then
  CMD+=(RESUME="${MTL_MLP_RESUME}")
fi

echo "[train] config=${MTL_MLP_CONFIG}"
if [[ -n "${MTL_MLP_RESUME}" ]]; then
  echo "[train] resume=${MTL_MLP_RESUME}"
fi

"${CMD[@]}"
