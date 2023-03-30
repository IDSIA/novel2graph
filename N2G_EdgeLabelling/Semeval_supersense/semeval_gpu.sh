#!/bin/bash
#SBATCH -J gpu_test
#SBATCH --ntasks=1
#SBATCH --partition=debug-gpu
#SBATCH --time=04:00:00
#SBATCH --mem=12020
#SBATCH --gres=gpu:1

# enable conda and activate environment
#. ~/apps/spack_install/linux-centos8-haswell/gcc-10.1.0/anaconda3-2019.10-s5gxujgmyc6xkbi6himjgi5hktpvcbin/etc/profile.d/conda.sh
#conda activate nlp

python Supersense_semeval.py

