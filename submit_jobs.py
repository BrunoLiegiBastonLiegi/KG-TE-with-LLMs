import json, argparse, os

with open('model_2_slurm_conf.json', 'r') as f:
    model2conf = json.load(f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Submit slurm jobs.')
    parser.add_argument('models', nargs='+')
    parser.add_argument('--data')
    parser.add_argument('--prompt')
    parser.add_argument('--groundtruth', action='store_true')
    parser.add_argument('--top_k', default=10)
    args = parser.parse_args()

    py_script = "python benchmark.py --data $1 --groundtruth" if args.groundtruth else "python benchmark.py --data $1 --conf $2 --prompt $3 --kb $4 --top_k 10"
    
    slurm_script = """#!/bin/bash
#SBATCH --job-name=benchmark_{model}_{data}         # Job name
#SBATCH --mail-type=END,FAIL          # Mail events
#SBATCH --mail-user=andrea.papaluca@anu.edu.au     # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem={mem}                    # Job memory request
#SBATCH --partition=gpu
#SBATCH --gres={gres}
#SBATCH --output=log/benchmark_{model}_{data}.log   # Standard output and error log
#SBATCH --time=72:00:00               # Time limit hrs:min:sec
pwd; hostname; date
    
echo "---- Running Benchmark ----"
source activate llama
{py_script}
echo "---- Done ----"
date
"""

    if args.data is None:
        data = ['webnlg/test.json', 'webnlg_modified/test.json', 'nyt/test.json']
        
    else:
        data = [data]

    for d in data:
        data_dir = d.split('/')[-2]
        kb = f"{data_dir}/kb_single_triples_normalized/"
        for model in args.models:
            name = model[:-5]
            slurm_conf = model2conf[name]
            with open('benchmark.slurm', 'w') as f:
                f.write(
                    slurm_script.format(
                        mem=slurm_conf['mem'],
                        gres=slurm_conf['gres'],
                        model=name,
                        data=data_dir,
                        py_script=py_script
                    )
                )
            command = f"sbatch benchmark.slurm {d} {model} {args.prompt} {kb} {args.top_k}"
            print(f'> Submitting job:\n  > {command}')
            os.system(command)
            if args.groundtruth: # avoid multiple evaluation of groundtruth
                break
