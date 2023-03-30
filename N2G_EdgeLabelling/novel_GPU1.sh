#!/bin/bash
#SBATCH -J gpu_novel_test
#SBATCH --ntasks=1
#SBATCH --partition=debug-gpu
#SBATCH --time=04:00:00
#SBATCH --output=simulation-m-%j.out
#SBATCH --error=simulation-m-%j.err
#SBATCH --mem=12020
#SBATCH --gres=gpu:1


#python test_gpu.py
#source activate ~/ENV_NLP1/bin/activate/
#python sample_GPU.py
python test_relations_clustering.py hp2.txt
