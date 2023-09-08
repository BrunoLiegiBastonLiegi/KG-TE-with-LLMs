import json
from utils import get_data_loader, normalize_triple
from tqdm import tqdm


if __name__ == '__main__':
    for dataset in ('webnlg', 'webnlg_modified', 'nyt'):
        ent2id = set()
        pred2id = set()
        for split in ('train', 'valid', 'test'):
            data = get_data_loader(f'{dataset}/{split}.json')
            for _, triples in tqdm(data, total=len(data)):
                if dataset == 'nyt':
                    triples = [ (t[0], t[1].split('/')[-1], t[2]) for t in triples ]
                for t in triples:
                    t = normalize_triple(t)
                    ent2id.add(t[0])
                    ent2id.add(t[2])
                    pred2id.add(t[1])
        ent2id = dict(zip(ent2id, range(len(ent2id))))
        pred2id = dict(zip(pred2id, range(len(pred2id))))
        with open(f'{dataset}/ent2id.json', 'w') as f:
            json.dump(ent2id, f, indent=2)
        with open(f'{dataset}/pred2id.json', 'w') as f:
            json.dump(pred2id, f, indent=2)
