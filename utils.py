import json

def get_triples_from_json(infile):
    sent2triple = {}
    with open(infile, 'r') as f:
        d = json.load(f)['entries']
        for v in d:
            v = list(v.values())[0]
            triples = [
                (t['subject'], t['property'], t['object'])
                for t in v['modifiedtripleset']
            ]
            for lex in v['lexicalisations']:
                sent2triple.update({
                    lex['lex'] : triples
                })

    triples = []
    for t in sent2triple.values():
        triples += t
    triples = set(triples)

    return sent2triple, triples



if __name__ == '__main__':
    import sys
    d, triples = get_triples_from_json(sys.argv[1])
    for k,v in d.items():
        for t in v:
            assert t in triples
        print('---------------------------------------------------------------')
        print(k)
        print(v)
        print('---------------------------------------------------------------')
