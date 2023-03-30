#!/bin/bash -l
#SBATCH --job-name=nlp_job
#SBATCH --time=48:00:00
#SBATCH --output=simulation-m-%j.out
#SBATCH --error=simulation-m-%j.err
#SBATCH --nodes=1
#SBATCH --mem=64000
source activate ~/ENV_NLP1/bin/activate/
python test_relations_clustering.py hp2.txt
