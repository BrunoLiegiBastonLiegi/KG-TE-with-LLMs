import matplotlib.pyplot as plt
import json, os, re
import numpy as np
from scipy.optimize import curve_fit

if __name__ == '__main__':

    #models = set('gpt2', 'gpt2-xl', 'falcon-7b', 'falcon-40b', 'llama-13b', 'llama-65b', 'gpt-3.5-turbo', 'gpt-4')

    with open ('model_to_n-params.json', 'r') as f:
        model_to_nparams = json.load(f)
    
    files = [ f
        for f in os.listdir('./')
        if re.search('performance_summary_.+\.json', f) is not None
    ]

    performance = {'WebNLG': {}, 'NYT': {}}
    for file in files:
        with open(file, 'r') as f:
            models, perfs = zip(*json.load(f).items())
        models = [ m.split(' ')[0] for m in models ]
        perfs = np.asarray([ list(p.values()) for p in perfs ])
        if 'webnlg_modified' in file:
            dataset = 'WebNLG'
            setting = re.search('(?<=performance_summary_).+(?=_webnlg_modified)', file).group(0)
        elif 'nyt' in file:
            dataset = 'NYT'
            setting = re.search('(?<=performance_summary_).+(?=_nyt)', file).group(0)
        performance[dataset][setting] = (models, perfs)

    for setting in ('zero-shot', 'zero-shot+kb', 'few-shots'):
         nparams = [
             model_to_nparams[model]
             for dataset in ('WebNLG', 'NYT')
             for model in performance[dataset][setting][0]
         ]
         f1 = np.concatenate(
         (performance['WebNLG'][setting][1][:,2],
          performance['NYT'][setting][1][:,2])
         )
         plt.scatter(nparams, f1, marker='*')

    plt.xscale('log')
    plt.show()
    
