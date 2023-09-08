from utils import normalize_triple, get_data_loader
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os, json
from utils import load_kb, get_llm, get_relevant_triples, triple_equality
import argparse


def triples_to_id(triples):
    ids = []
    for t in triples:
        ids.append(
            [ ent2id[t[0]], pred2id[t[1]], ent2id[t[2]] ]
        )
    return np.asarray(ids)

def stats_gen(dataset):
    
    n_tokens_hist = {'test': {}, 'train': {}}
    n_triples_hist = {'test': {}, 'train': {}}
    relations_hist = {'test': {}, 'train': {}}
    entities_hist = {'test': {}, 'train': {}}
    top_k_to_n_matches = {
        'standard': dict(zip([3, 5, 10, 20, 50], [0, 0, 0, 0, 0])),
        'complete': dict(zip([3, 5, 10, 20, 50], [0, 0, 0, 0, 0])),
    }
    _, service_context = get_llm('gpt2', 'text-generation', max_new_tokens=8, temperature=0.1, load_in_8bit=True)
    kb_path = f'{dataset}/kb_single_triples_normalized'
    retrievers = {
        'standard': {
            i: load_kb(kb_path, service_context, i)[1]
            for i in top_k_to_n_matches['standard'].keys()
        },
        'complete': {
            i: load_kb(kb_path + '_complete', service_context, i)[1]
            for i in top_k_to_n_matches['complete'].keys()
        }
    }
    total_number_of_triples = 0
    
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
            if dataset == 'nyt':
                triples = [ (t[0], t[1].split('/')[-1], t[2]) for t in triples ]
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
            if split == 'test':
                total_number_of_triples += len(triples)
                for case in ('standard', 'complete'):
                    for i, retriever in retrievers[case].items():
                        relevant_triples = set(get_relevant_triples(sentence, retriever, return_tuple=True))
                        #if i == 3:
                        #    print(f'True Triples:\n{triples}')
                        #    print(f'Context Triples:\n{relevant_triples}')
                        relevant_triples = triples_to_id(relevant_triples)
                        triples_ids = triples_to_id(triples)
                        for t in triples_ids:
                            count = len((relevant_triples == t).all(-1).nonzero()[0])
                            if count == 1:
                                top_k_to_n_matches[case][i] += 1
                            elif count == 0:
                                continue
                            else:
                                print('> Warning')
                                print(f'Retrieved Triplets:\n{relevant_triples}')
                                print(f'True Triplet:\n{t}')
                                #raise AssertionError('Incompatible matching of triplets.')
                                continue

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

    plt.rcParams.update({'font.size': 24})

    print('--> TOT: ', total_number_of_triples)
    # calculate overlapping between test and train triples
    from utils import get_data_from_files
    _, train_triples = get_data_from_files(f'{dataset}/train.json')
    _, valid_triples = get_data_from_files(f'{dataset}/valid.json')
    _, test_triples = get_data_from_files(f'{dataset}/test.json')
    train_triples = set([tuple(normalize_triple(t)) for t in train_triples])
    valid_triples = set([tuple(normalize_triple(t)) for t in valid_triples])
    test_triples = set([tuple(normalize_triple(t)) for t in test_triples])
    overlap = train_triples.union(valid_triples).intersection(test_triples)
    overlap = len(overlap)/len(test_triples)
    print(top_k_to_n_matches)
    for case in ('standard', 'complete'):
        top_k, n_matches = zip(*top_k_to_n_matches[case].items())
        n_matches = [ n/total_number_of_triples for n in n_matches ]
        print(n_matches)
        #plt.scatter(top_k, n_matches)
        label = 'train + valid'
        if case == 'complete':
            label += ' + test'
        plt.plot(top_k, n_matches, markersize=15, linewidth=2, marker='.', label=label)

    plt.axhline(y=overlap, c='black', linestyle='--', linewidth=2)
    plt.ylabel('Probability of Finding the True Triplet')
    plt.xlabel('Number of Context Triplets Retrieved')    
    plt.tight_layout()
    figname = f'n-matches_vs_top-k_{dataset}.pdf'
    plt.legend()
    plt.savefig(figname, format='pdf')
    plt.show()

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

    print(n_triples_hist)
    # number of triples per sentence histogram
    for split in ('train', 'test'):
        hist, bins = n_triples_hist[split]
        try:
            width = 1 * (bins[1] - bins[0])
        except:
            width = 1
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

    print(top_k_to_n_matches)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='KB Construction.')
    parser.add_argument('--complete', action='store_true')
    args = parser.parse_args()

    
    datasets = ['webnlg_modified', 'webnlg', 'nyt']
    for dataset in datasets:
        with open(f"{dataset}/ent2id.json", 'r') as f:
            ent2id = json.load(f)
        with open(f"{dataset}/pred2id.json", 'r') as f:
            pred2id = json.load(f)
        stats_gen(dataset)
