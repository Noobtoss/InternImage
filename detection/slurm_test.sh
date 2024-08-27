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

BASE_DIR=/nfs/scratch/staff/schmittth/sync/InternImage/detection
CONFIG=$1
CHECKPOINT=$2

srun python -u test.py $BASE_DIR/$CONFIG $BASE_DIR/$CHECKPOINT --launcher='slurm' ${@:3} --eval 'bbox' --out "test.pkl"
