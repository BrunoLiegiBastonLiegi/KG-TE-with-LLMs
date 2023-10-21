import matplotlib.pyplot as plt
import json, os
import numpy as np
from scipy.optimize import curve_fit

def get_P_Nkb(dataset, few_shots=False):
    
    scales = [0.5, 0.25, 0.1]
    P_Nkb = {}
    name = f'{dataset}/P_Nkb'
    if few_shots:
        name += '_few-shots'
    with open(f'{name}.json', 'r') as f:
        Nkb, P = list(zip(*json.load(f).items()))
        Nkb = [ float(n) for n in Nkb ] 
        P_Nkb[1.] = (Nkb, P)
    for scale in scales:
        try:
            with open(f'{name}_scale-{scale}.json', 'r') as f:
                Nkb, P = list(zip(*json.load(f).items()))
                Nkb = [ float(n) for n in Nkb ] 
                P_Nkb[scale] = (Nkb, P)
        except:
            P_Nkb[scale] = None
            print(f'> Scale {scale} for {dataset} not found, skipping.')
    return P_Nkb

def get_random_comparison(dataset, few_shots=False):

    filename = f'{dataset}/random_vs_llama65b'
    if few_shots:
            filename += '_few-shots'
    else:
        filename += '_zero-shot'
    filename += '.json'
    with open(filename, 'r') as f:
        d = json.load(f)

    random_f = lambda n: n
    
    for model, perf in d.items():
        Nkb, F1 = list(zip(*perf.items()))
        Nkb = [ float(n) for n in Nkb ]
        d[model] = (Nkb, F1)

    return d

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Data statistics.')
    parser.add_argument('datasets', nargs='*')
    parser.add_argument('--few_shots', action='store_true')
    args = parser.parse_args()

        
    
    datasets = args.datasets
    if len(datasets) == 0:
        datasets = ['webnlg_modified', 'nyt']

    plt.rcParams.update({'font.size': 24})
        
    fig = plt.figure(figsize=(10,10))
    P_Nkb = {}
    P_Nkb_fs = {}
    for dataset in datasets:
        P_Nkb[dataset] = get_P_Nkb(dataset)
        P_Nkb_fs[dataset] = get_P_Nkb(dataset, few_shots=True)

    P = P_Nkb_fs if args.few_shots else P_Nkb
        
    plt.rcParams.update({'font.size': 24})
        
    fig = plt.figure(figsize=(10,10))

    for dataset in datasets:
        label = 'WebNLG' if dataset == 'webnlg_modified' or dataset == 'webnlg' else 'NYT'
        plt.plot(P[dataset][1.][0], P[dataset][1.][1], label=label, linewidth=2, marker='.', markersize=15)
        
        # fit the P(Nkb) with a 1/x shape
        P_f = lambda n, a, b: - a/ n + b
        Nkb, prob = P[dataset][1.]
        #plt.plot(Nkb,P, marker='.')
        P_pars, _ = curve_fit(P_f, Nkb, prob)
        Nkb = np.linspace(min(Nkb), max(Nkb), num=100)
        plt.plot(Nkb, [P_f(n, *P_pars) for n in Nkb])
        
    plt.xlabel(r'$N_{KB}$')
    plt.ylabel(r'$P$')
    plt.legend()
    savefile = 'P_Nkb'
    if args.few_shots:
        savefile += '_few-shots'
    savefile += '.pdf'
    plt.savefig(savefile, format='pdf', dpi=300)
    plt.show()
    
    for dataset in datasets:
        
        plt.figure(figsize=(10,10))
        
        for scale, element in P[dataset].items():
            if element is not None:
                Nkb, prob = element
                plt.plot(Nkb, prob, label=rf'$S = {scale}$', linewidth=2, marker='.', markersize=15)
            
        plt.xlabel(r'$N_{KB}$')
        plt.ylabel(r'$P$')
        plt.legend()
        plt.savefig('P_Nkb_different_scales.pdf', format='pdf', dpi=300)
        plt.show()

        # fit the P(Nkb) with a 1/x shape
        P_f = lambda n, a, b: - a/ n + b
        Nkb, prob = P[dataset][1.]
        plt.plot(Nkb, prob, marker='.')
        P_pars, _ = curve_fit(P_f, Nkb, prob)
        Nkb = np.linspace(1, max(Nkb), num=100)
        plt.plot(Nkb, [P_f(n, *P_pars) for n in Nkb])
        plt.show()

        plt.figure(figsize=(10,10))

        N = 1
        random_f = lambda n, a, b, c: a * np.power(P_f(n, *P_pars)/n, b*N) + c
        d = get_random_comparison(dataset, few_shots=args.few_shots)
        model_name = {'random': 'Random', 'llama65b':'LLaMA 65b'}
        for model, (Nkb, F1) in d.items():
            plt.plot(Nkb, F1, label=model_name[model], linewidth=2, marker='.', markersize=15)
            if model == 'random': # and not args.few_shots:
                F1_pars, _ = curve_fit(random_f, Nkb, F1, maxfev=5000)
                print(f'> Optimal F1 parameters: {F1_pars}')
                Nkb = np.linspace(min(Nkb), max(Nkb), num=100)
                F1 = [ random_f(n, *F1_pars) for n in Nkb ]
                plt.plot(Nkb, F1, label=r'$\alpha\cdot\left(\frac{P(N_{KB})}{N_{KB}}\right)^n+\beta$', linewidth=2, linestyle='--')
                            
        plt.xlabel(r'$N_{KB}$')
        plt.ylabel(r'$F1$')
        plt.legend()
        savename = 'random_f1'
        if args.few_shots:
            savename += '_few-shots'
        savename += '.pdf'
        plt.savefig(savename, format='pdf', dpi=300)
        plt.show()


    plt.figure(figsize=(10,10))

    llama65_zs_perf = {0: 0.219, 0.1: 0.210, 0.25: 0.243, 0.5: 0.299, 1: 0.372}
    llama65_fs_perf = {0: 0.219, 0.1: 0.548, 0.25: 0.608, 0.5: 0.639, 1: 0.677}
    
    for P, perf, setting in zip((P_Nkb, P_Nkb_fs), (llama65_zs_perf, llama65_fs_perf), ('Zero-Shot', 'Few-Shots')):
        probs = {0: 0}
        print(P)
        for scale, element in P['webnlg_modified'].items():
            if element is not None:
                Nkb, P = element
                idx = Nkb.index(5)
                probs[scale] = P[idx]
        probs = [ probs[scale] for scale in perf.keys() ]
        perf = list(perf.values())
        marker = '^' if setting == 'Few-Shots' else 'v'
        plt.scatter(probs, perf, s=50, label=setting, marker=marker)
        fit_pars = np.polyfit(probs, perf, deg=1)
        line = lambda x, a, b: a*x + b
        space = np.linspace(0,1)
        plt.plot(space, line(space, *fit_pars), linestyle='--', linewidth=2)

    plt.xlabel(r'$P_{S}(N_{KB}=5)$')
    plt.ylabel(r'$F1$')
    plt.legend()
    plt.savefig('F1_vs_scale.pdf', format='pdf', dpi=300)
    plt.show()        

    
