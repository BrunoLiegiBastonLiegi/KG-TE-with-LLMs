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
    fig = plt.figure(figsize=(17,8))
    ax = plt.subplot(111)
    
    marker = dict(zip(('zero-shot', 'zero-shot+kb', 'few-shots'), ('o','*','v')))
    label = dict(zip(('zero-shot', 'zero-shot+kb', 'few-shots'), ('Zero-Shot','Zero-Shot + KB','Few-Shots')))
    scatter = []
    kb_scatter = []
    fs_scatter = []
    label = {'zero-shot': '2-Shots', 'zero-shot+kb': '0.5-Shots (KB)', 'few-shots': '5-shots (KB)'}
    fit = lambda x,a,b: a * x + b #lambda x,a,b: a*np.log(x) + b
    for setting in ('zero-shot', 'zero-shot+kb', 'few-shots'):
         nparams = [
             model_to_nparams[model]
             for dataset in ('WebNLG', 'NYT')
             for model in performance[dataset][setting][0]
         ]
         if setting != 'zero-shot':
             indices = (
                 performance[dataset][setting][0].index('gpt-4') + 8,
                 performance[dataset][setting][0].index('gpt-3.5-turbo') + 8
             )
         else:
             indices = ()
         
         if setting == 'zero-shot':
             color = ["#1f77b4" for _ in range(len(nparams))]
         elif setting == 'zero-shot+kb':
             color = ["#ff7f0e" for _ in range(len(nparams))]
         else:
             color = ["#2ca02c" for _ in range(len(nparams))]
         for i in indices:
             color[i] = "grey"
         web_perf = performance['WebNLG'][setting][1][:,2]
         nyt_perf = performance['NYT'][setting][1][:,2]
         f1 = np.concatenate(
         (web_perf / web_perf.max(),
          nyt_perf / nyt_perf.max())
         )
         nparams = np.log(nparams)
         ax.scatter(nparams, f1, s=200, label=label[setting], c=color)
         x = nparams
         if setting == 'zero-shot':
             for i,item in enumerate(zip(nparams, f1)):
                 scatter.append(item)
         elif setting == 'zero-shot+kb':
             for i,item in enumerate(zip(nparams, f1)):
                 if i not in indices:
                     kb_scatter.append(item)
         else:
             for i,item in enumerate(zip(nparams, f1)):
                 if i not in indices:
                     fs_scatter.append(item)
         #pars, _ = curve_fit(fit, x, f1)
         #space = np.linspace(min(x), max(x), num=100)
         #plt.plot(space, fit(space, *pars), linewidth=5)
    def get_r2(y_pred, y_true):
        ss_res = ((y_pred - y_true)**2).sum()
        ss_tot = ((y_true - y_true.mean())**2).sum()
        return 1 - (ss_res/ss_tot)

    for scat in (scatter, kb_scatter, fs_scatter):
        x, y = zip(*scat)
        space = np.linspace(min(x), max(x), num=100)
        pars = np.polyfit(x, y, 1)#curve_fit(fit, x, y)
        print(pars)
        print(get_r2(fit(np.array(x), *pars), np.array(y)))
        ax.plot(space, fit(space, *pars), linewidth=5, linestyle='--')
    #plt.xscale('log')
    plt.xlabel(r'$N$ Parameters')
    plt.ylabel('Normalized $F1$')
    box = ax.get_position()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
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
            f1 = f1 / f1.max()
            plt.scatter(nparams, f1, marker=marker[setting], s=2000, label=label[setting])

        plt.xscale('log')
        plt.xlabel(r'$N$ Parameters')
        plt.ylabel('Normalized $F1$')
        #plt.legend()
        #plt.show()
        plt.tight_layout()
        plt.savefig(f'f1_vs_nparams_{dataset}.pdf', format='pdf', dpi=300)
        
