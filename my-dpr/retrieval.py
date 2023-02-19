import json
import os

from utils import Searcher, Encoder
from datasets import load_dataset
from tqdm import tqdm
import datetime


with open('config/retrieval.json') as f:
    config = json.load(f)
dev_data = load_dataset('csv', data_files=config['dev_file'], delimiter='\t')['train']
encoder = Encoder(config['encoder_path'])
searcher = Searcher(encoder, config['index_file'], config['topK'], config['nlist'], config['nprobe'])

print('------现在开始检索------')

outfile = open(config['results_dir'] + os.sep + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.json', mode='w')

for i in tqdm(range(len(dev_data))):
    qid = dev_data[i]['qid']
    query = dev_data[i]['content']
    scores, indices = searcher.search(query)
    for score, index in zip(scores, indices):
        outfile.write('\t'.join([str(qid), str(index), str(score)]).strip() + '\n')

outfile.close()
