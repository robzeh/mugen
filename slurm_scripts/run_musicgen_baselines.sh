#!/bin/sh

ln -sfn /checkpoints/${USER}/${SLURM_JOB_ID} $PWD/checkpoint

# touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE

module purge
module load python/3.9.10
source /h/robzeh/projects/mugen/clap_venv/bin/activate
# python -m venv mugen_venv2
# source mugen_venv/bin/activate
python -m pip install -r requirements2.txt

python generate_baselines.py
