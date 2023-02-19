from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import json

with open('config/read_rels.json') as f:
    config = json.load(f)


checkpoint_name = 'bert-base-multilingual-cased'
rels = load_dataset('csv', data_files=config['rels_file'], delimiter='\t')['train']
queries = load_dataset('csv', data_files=config['query_file'], delimiter='\t')['train']
corpus = load_dataset('csv', data_files=config['corpus_file'], delimiter='\t')['train']
tokenizer = AutoTokenizer.from_pretrained(checkpoint_name, use_fast=True)
qids = np.array(queries['qid'])
pids = np.array(corpus['pid']) # 方便定位passage的index,np查找更快
rel_pids = np.array(rels['pid'])

qps = []

for i in tqdm(range(len(rels))):
    qid = rels[i]['qid']
    pid = rels[i]['pid']
    q_pos = np.where(qids == qid)
    p_pos = np.where(pids == pid)
    query = queries[q_pos]['content'][0]
    passage = corpus[p_pos]['content'][0]
    # span = [query, passage]
    # tokenized_span = [
    #     tokenizer(
    #         s.lower(),
    #         add_special_tokens=True,
    #         truncation=True,
    #         max_length=128,
    #         return_attention_mask=False,
    #         return_token_type_ids=False,
    #     )['input_ids'] for s in span
    # ]
    # all_tokenized.append(json.dumps({'tokenized_query':tokenized_span[0], 'tokenized_passage':tokenized_span[1]}))
    qps.append(json.dumps({'query':query, 'passage':passage}))


with open(config['out_file'], mode='w', encoding='utf-8') as f:
    for x in qps:
        if x is None:
            continue
        f.write(x + '\n')





'''
passage retrieval任务不能把query和passage同时批量处理，不然到后面神经网络那里会无法处理
'''
# def token_fn(example):
#     batch_qids = np.array(example['qid'])
#     batch_pids = np.array(example['pid']) # 一堆pid
#     q_pos = np.where(qids == batch_qids[:, None])[-1]
#     p_pos = np.where(pids == batch_pids[:, None])[-1] # 注意这个方法是写死的，会返回pid的index
#     batch_queries = queries[q_pos]['content']
#     batch_passages = corpus[p_pos]['content'] # 直接用一堆index去找passage
#     return tokenizer(batch_queries, batch_passages, max_length=128, truncation=True, padding=True, return_tensors='pt')
#
# tokenized_datasets = rels.map(token_fn, batched=True) # 批量使用上面的函数
# tokenized_datasets = tokenized_datasets.remove_columns(['qid', '0', 'pid', '1']) # 移除不需要的column
# tokenized_datasets.set_format('torch') # 设置list为tensor
# tokenized_datasets.save_to_disk('../data/medical/tokenized_dataset')
