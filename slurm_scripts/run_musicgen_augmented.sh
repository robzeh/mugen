#!/bin/sh

ln -sfn /checkpoints/${USER}/${SLURM_JOB_ID} $PWD/checkpoint
# touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE

module purge
module load python/3.9.10
# /pkgs/python-3.10.12/bin/python3 -m venv clap_venv
source /h/robzeh/projects/raptune/raptune_venv/bin/activate
python -m pip install -r /h/robzeh/projects/raptune/requirements.txt

python generate_qualities_augmented.py

