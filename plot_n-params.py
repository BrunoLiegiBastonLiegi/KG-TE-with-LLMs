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

    plt.rcParams.update({'font.size': 44})
    plt.figure(figsize=(10,8))
    
    marker = dict(zip(('zero-shot', 'zero-shot+kb', 'few-shots'), ('o','*','v')))
    label = dict(zip(('zero-shot', 'zero-shot+kb', 'few-shots'), ('Zero-Shot','Zero-Shot + KB','Few-Shots')))
    fit = lambda x,a,b: a*np.log(x) + b
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
         plt.scatter(nparams, f1, marker=marker[setting], s=200, label=label[setting])
         x = nparams
         if setting == 'few-shots':
             indices = (f1 < 0.2).nonzero()[0]
             print(indices)
             f1 = f1[f1 > 0.2]
             print(f1)
             for i in indices:
                 del x[i]
         pars, _ = curve_fit(fit, x, f1)
         space = np.linspace(min(x), max(x), num=100)
         plt.plot(space, fit(space, *pars), linewidth=5)
         
    plt.xscale('log')
    plt.xlabel(r'$N$ Parameters')
    plt.ylabel('$F1$')
    #plt.legend()
    plt.tight_layout()
    plt.savefig('f1_vs_nparams.pdf', format='pdf', dpi=300)
    plt.show()
    
    for dataset in ('WebNLG', 'NYT'):
        
        plt.figure(figsize=(10,8))
        
        for setting in ('zero-shot', 'zero-shot+kb', 'few-shots'):
            nparams = [
                model_to_nparams[model]
                for model in performance[dataset][setting][0]
            ]
            f1 = performance[dataset][setting][1][:,2]
            plt.scatter(nparams, f1, marker=marker[setting], s=2000, label=label[setting])

        plt.xscale('log')
        plt.xlabel(r'$N$ Parameters')
        plt.ylabel('$F1$')
        #plt.legend()
        plt.show()
    
