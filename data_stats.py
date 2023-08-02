from utils import normalize_triple, get_data_loader
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

def stats_gen(dataset):
    
    n_tokens_hist = {'test': {}, 'train': {}}
    n_triples_hist = {'test': {}, 'train': {}}
    relations_hist = {'test': {}, 'train': {}}
    entities_hist = {'test': {}, 'train': {}}
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    for split in ('train', 'test'):
        
        if split == 'train':
            train = get_data_loader(f'{dataset}/train.json')
            valid = get_data_loader(f'{dataset}/valid.json')
            data = list(train) + list(valid)
        else:
            data = get_data_loader(f'{dataset}/test.json')

        sentences, n_triples, relations, entities = [], [], [], []
        for sentence, triples in tqdm(data, total=len(data)):
            triples = [normalize_triple(t) for t in triples]
            sentences.append(sentence)
            # number of triples per sentence histogram
            n_triples.append(len(triples))
            # relation types histogram
            relations += [os.path.basename(t[1]) for t in triples]
            # entities histogram
            for t in triples:
                entities.append(t[0])
                entities.append(t[2])      

        # number of tokens per sentence histogram
        tok_sents = tokenizer(text=sentences, padding=False)
        n_tokens = [ len(tokens) for tokens in tok_sents.input_ids ]
        n_tokens_hist[split] = np.histogram(n_tokens, density=True)
        # number of triples per sentence histogram
        n_triples_hist[split] = np.histogram(n_triples, bins=range(1, max(n_triples)+1), density=True)
        # relation types histogram
        relations = sorted(Counter(relations).items(), key=lambda x: x[1], reverse=True)
        relations = list(zip(*relations))
        relations_hist[split] = (list(relations[0]), list(relations[1]))
        # entities histogram
        entities = sorted(Counter(entities).items(), key=lambda x: x[1], reverse=True)
        entities = list(zip(*entities))
        entities_hist[split] = (list(entities[0]), list(entities[1]))

    fig, axes = plt.subplots(1,2, figsize=(12,6))
    # number of tokens per sentence histogram   
    for split in ('train', 'test'):
        hist, bins = n_tokens_hist[split]
        width = 1 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        label = split if split == 'test' else 'train + validation'
        axes[0].bar(center, hist, align='center', width=width, alpha=0.5, label=label)
    axes[0].set_xlabel('Number of Tokens per Sentence')
    axes[0].legend()

    # number of triples per sentence histogram
    for split in ('train', 'test'):
        hist, bins = n_triples_hist[split]
        width = 1 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        label = split if split == 'test' else 'train + validation'
        axes[1].bar(center, hist, align='center', width=width, alpha=0.5, label=label)
    axes[1].set_xlabel('Number of Triplets per Sentence')
    axes[1].legend()
    fig.tight_layout()
    plt.savefig(f'n_tokens+triples_per_sentence_{dataset}.pdf', format='pdf')
    plt.clf()

    fig, axes = plt.subplots(2,1, figsize=(12,12))
    # relation types histogram
    relations = set(relations_hist['train'][0] + relations_hist['test'][0])
    for rel in relations:
        if rel not in relations_hist['train'][0]:
            relations_hist['train'][0].append(rel)
            relations_hist['train'][1].append(0)
        elif rel not in relations_hist['test'][0]:
            relations_hist['test'][0].append(rel)
            relations_hist['test'][1].append(0)
    height = relations_hist['train'][1]/np.sum(relations_hist['train'][1])
    axes[0].bar(relations_hist['train'][0], height, alpha=0.5, label='train + validation')
    #plt.xticks(rotation=90, fontsize='xx-small')
    height = relations_hist['test'][1]/np.sum(relations_hist['test'][1])
    axes[0].bar(relations_hist['test'][0], height, alpha=0.5, label='test')
    axes[0].get_xaxis().set_ticks([])
    axes[0].set_xlabel('Relations')
    axes[0].legend()

    # entities histogram
    entities = set(relations_hist['train'][0] + relations_hist['test'][0])
    for e in entities:
        if e not in entities_hist['train'][0]:
            entities_hist['train'][0].append(e)
            entities_hist['train'][1].append(0)
        elif e not in entities_hist['test'][0]:
            entities_hist['test'][0].append(e)
            entities_hist['test'][1].append(0)
    height = entities_hist['train'][1]/np.sum(entities_hist['train'][1])
    axes[1].bar(entities_hist['train'][0], height, alpha=0.5, label='train + validation')
    #axes[1].xticks(rotation=90, fontsize='xx-small')
    height = entities_hist['test'][1]/np.sum(entities_hist['test'][1])
    axes[1].bar(entities_hist['test'][0], height, alpha=0.5, label='test')
    axes[1].legend()
    axes[1].get_xaxis().set_ticks([])
    axes[1].set_xlabel('Entities')

    fig.tight_layout()
    plt.savefig(f'relations+entities_distribution_{dataset}.pdf', format='pdf')
    plt.clf()
    

if __name__ == '__main__':

    import argparse

    datasets = ['webnlg', 'nyt', 'webnlg_modified']
    for dataset in datasets:
        stats_gen(dataset)
