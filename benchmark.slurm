#!/bin/bash
#SBATCH --job-name=benchmark_gpt2_nyt         # Job name
#SBATCH --mail-type=END,FAIL          # Mail events
#SBATCH --mail-user=andrea.papaluca@anu.edu.au     # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=16gb                    # Job memory request
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=log/benchmark_gpt2_nyt.log   # Standard output and error log
#SBATCH --time=72:00:00               # Time limit hrs:min:sec
pwd; hostname; date
    
echo "---- Running Benchmark ----"
source activate llama
python benchmark.py --data $1 --conf $2 --prompt $3
echo "---- Done ----"
date
