#!/bin/bash
#SBATCH --job-name=kb_building         # Job name
#SBATCH --mail-type=END,FAIL          # Mail events
#SBATCH --mail-user=andrea.papaluca@anu.edu.au     # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=110gb                    # Job memory request
#SBATCH --partition=bigdata
#SBATCH --qos=bigdata
#SBATCH --gres=gpu:1
#SBATCH --output=kb_builder.log   # Standard output and error log
#SBATCH --time=72:00:00               # Time limit hrs:min:sec
pwd; hostname; date

echo "---- Building KB ----"
source activate llama
#python -m cProfile benchmark.py
python kb_builder.py --data $* --normalize
echo "---- Done ----"
date
