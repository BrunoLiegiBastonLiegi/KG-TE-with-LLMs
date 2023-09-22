import matplotlib.pyplot as plt
import json, os

def get_P_Nkb(dataset):
    
    scales = [0.5, 0.25]
    P_Nkb = {}
    name = f'{dataset}/P_Nkb'
    if args.few_shots:
        name += '_few-shots'
    with open(f'{name}.json', 'r') as f:
        P_Nkb[1.] = list(zip(*json.load(f).items()))
    for scale in scales:
        try:
            with open(f'{name}_scale-{scale}.json', 'r') as f:
                P_Nkb[scale] = list(zip(*json.load(f).items()))
        except:
            P_Nkb[scale] = None
            print(f'> Scale {scale} for {dataset} not found, skipping.')
    return P_Nkb


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Data statistics.')
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('--few_shots', action='store_true')
    args = parser.parse_args()

        
    
    datasets = args.datasets
    if len(datasets) == 0:
        datasets = ['webnlg_modified', 'nyt']

    plt.rcParams.update({'font.size': 24})
        
    fig = plt.figure(figsize=(10,10))
    P_Nkb = {}
    for dataset in datasets:
        P_Nkb['dataset'] = get_P_Nkb(dataset)

    plt.rcParams.update({'font.size': 24})
        
    fig = plt.figure(figsize=(10,10))

    for dataset in datasets:
        label = 'WebNLG' if dataset == 'weblg_modified' or dataset == 'webnlg' else 'NYT'
        plt.plot(P_Nkb[dataset][1.][0], P_Nkb[dataset][1.][1], label=rf'$dataset$', linewidth=2, marker='.', markersize=15)
        
    fig.xlabel(r'$N_{KB}$')
    fig.ylabel(r'$P$')
    fig.legend()
    plt.savefig('P_Nkb.pdf', format='pdf', dpi=300)

    for dataset in datasets:
        
        plt.figure(figsize=(10,10))
        
        for scale, element in P_Nkb.items():
            if element is not None:
                Nkb, P = element
                plt.plot(Nkb, P, label=rf'$S = {scale}$', linewidth=2, marker='.', markersize=15)
            
        plt.xlabel(r'$N_{KB}$')
        plt.ylabel(r'$P$')
        plt.legend()
        plt.savefig('P_Nkb_different_scales.pdf', format='pdf', dpi=300)
        plt.show()

                
