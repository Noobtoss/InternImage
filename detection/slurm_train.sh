#!/bin/bash
#
#SBATCH --job-name=InternImage
#SBATCH --output=R-%j.out
#SBATCH --error=E-%j.err
#SBATCH --mail-user=thomas.schmitt@th-nuernberg.de
#SBATCH --mail-type=ALL
#
#SBATCH --partition=p2
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=64G                # RAM pro CPU Kern #20G #32G #64G

module purge
module load python/anaconda3
module load cuda/cuda-11.6.2
module load cudnn/cudnn-8.7.0.84-11.8
module load gcc/gcc-10.5.0
eval "$(conda shell.bash hook)"

conda activate InternImage

CONFIG=$1
VAULT_DIR=$2
#PORT=${PORT:-29300}
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

#mkdir $VAULT_DIR/training_logs/internimage/$SLURM_JOB_ID
#CACHE_DIR=$VAULT_DIR/.cache
#export PIP_CACHE_DIR=$CACHE_DIR
#export TRANSFORMERS_CACHE=$CACHE_DIR
#export HF_HOME=$CACHE_DIR
#mkdir -p $CACHE_DIR
#export TORCH_HOME=$VAULT_DIR/models/torchhub
#mkdir -p $TORCH_HOME

#srun python -u $VAULT_DIR/software/InternImage/segmentation/train.py $CONFIG --work-dir=$VAULT_DIR/training_logs/internimage/$SLURM_JOB_ID --launcher='slurm' ${@:3}
srun python -u train.py $CONFIG --launcher='slurm' ${@:3}
