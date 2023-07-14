#!/bin/bash
#SBATCH --job-name=benchmark         # Job name
#SBATCH --mail-type=END,FAIL          # Mail events
#SBATCH --mail-user=andrea.papaluca@anu.edu.au     # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=110gb                    # Job memory request
#SBATCH --partition=gpu
##SBATCH --gres=gpu:3090:1
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=benchmark.log   # Standard output and error log
#SBATCH --time=72:00:00               # Time limit hrs:min:sec
pwd; hostname; date

echo "---- Running WebNLG Benchmark ----"
source activate llama
python benchmark.py --data $1 --conf $2 --prompt $3 --kb $4 --top_k 10
#python benchmark.py --data $1 --groundtruth
echo "---- Done ----"
date