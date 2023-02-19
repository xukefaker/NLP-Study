import numpy as np

from utils import BertForMatching, Encoder
from transformers import BertTokenizer
from datasets import load_dataset
import json
import torch

if __name__ == '__main__':
    with open('config/encode_corpus.json') as f:
        config = json.load(f)

    encoder = Encoder(model_path=config['model_path'], pool_type=config['pool_type'],
                      max_seq_length=config['max_seq_length'], batch_size=config['batch_size'])
    encoder.encode_corpus(config['corpus_path'], config['output_path'])
