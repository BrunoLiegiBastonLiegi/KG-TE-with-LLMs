import argparse, json

parser = argparse.ArgumentParser(description='Get relevant triples from KB.')
parser.add_argument('--data', default='./webnlg-dataset_v3.0/corpus-reader/dev.json')
parser.add_argument('--kb', default='kb')
parser.add_argument('--conf', default='llm.conf')
args = parser.parse_args()
