import torch
from transformers import BertPreTrainedModel, BertModel, BertTokenizer
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset
import faiss
import numpy as np


# Dataloader会调用collator来获取一条条数据
class CollatorWithPaddingTruncation():
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def trunc(self, example):
        # 这里传入的example，是我们之前通过tokenizer处理获得的input_ids
        if len(example) < self.max_seq_length:
            return example
        else:
            return example[:self.max_seq_length]

    def pad(self, example, padding_num=0):
        # padding_num就是我们用来填充的数字
        assert len(example) < self.max_seq_length
        return example + [padding_num for i in range(self.max_seq_length - len(example))]

    def __call__(self, examples):
        # 这里的examples格式是自己写的
        # 此处的样例为：[{"query":xxxx, "passage":xxx}....]
        sentences = []  # 2个为1对，第一个是query，第二个是passage
        masks = []
        encoded_sentences = []  # 添加过特殊字符的词表表示
        for example in examples:
            for k, v in example.items():
                sentences.append(self.trunc(v))
        for sentence in sentences:
            encoded = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,  # 一定要先truncate，然后再添加特殊token，不然可能会出问题
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
            )
            masks.append(encoded['attention_mask'])
            encoded_sentences.append(encoded['input_ids'])
        batch = {
            'input_ids': torch.tensor(encoded_sentences),
            'attention_mask': torch.tensor(masks)
        }
        return batch


'''
我们选用的是最基本的Bert模型，它的最后一层输出是(batchsize, seq_length, hidden_size)
seq_length是我们在做分词的时候定义的max_length
hidden_size默认是768
对于单个seq，它的representation是(seq_length, hidden_size) --一个matrix
但是一个seq，我们必须要用一个vector来表示它，即把它处理成(1, hidden_size)
这个处理过程就是pooling，一般有三种选择：cls,max,avg
最后记得，让我们的rep_vector过一个线性层和一个非线性激活层
'''


class PoolingLayer(nn.Module):
    def __init__(self, pool_type='cls', input_size=768, output_size=768):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()
        self.pool_type = pool_type

    # forward重写后会代替__call__
    def forward(self, last_hidden_states, attention_mask):
        if self.pool_type == 'cls':  # 取第一个token的rep作为整个seq的rep
            rep_tensor = last_hidden_states[:,
                         0]  # last_hidden_states的shape是(batchsize, seq_length, hidden_size)，按理来说我们应该写成last_hidden_states[:, 0, :]，但实际上最后的:可以不用写
        elif self.pool_type == 'max':  # 取rep最大的token的rep作为整个seq的rep
            x = last_hidden_states * attention_mask.unsqueeze(-1)
            rep_tensor = F.max_pool1d(x.transpose(1, 2).contiguous(), x.size(1)).squeeze(-1)
        elif self.pool_type == "avg":
            rep_tensor = ((last_hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(
                -1))
        pooled_output = rep_tensor
        return pooled_output


class BertForMatching(BertPreTrainedModel):
    def __init__(self, config, pool_type="cls", add_pooling_layer=False, is_train=True, *model_args, **model_kargs):
        super().__init__(config)
        self.config = config
        self.is_train = is_train
        self.pooling_type = pool_type
        self.add_pooling_layer = add_pooling_layer

        self.bert = BertModel(config, add_pooling_layer)
        self.pooling_layer = PoolingLayer(self.pooling_type)

        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        '''
        outputs常用的属性：
        last_hidden_state
        pooler_output
        hidden_states
        见doc：https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/output
        '''
        seq_rep = outputs.last_hidden_state # shape: (batch_size, seq_length, hidden_size)
        pooled_seq_rep = self.pooling_layer(seq_rep, attention_mask)  # shape:(batch_size, hidden_size)，这个就是句子的最终表示
        loss = None
        if self.is_train:
            device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
            query_rep = pooled_seq_rep[0::2, ]  # 到这里可以直接通过dot来求相似度，但是由于直接乘出来的结果会很大，所以让每个vector除以一个范数再求相似度
            psg_rep = pooled_seq_rep[1::2, ]
            query_rep_norm = query_rep / query_rep.norm(dim=1)[:, None]  # 加一个[:, None]可以加一个维度，然后利用广播机制
            psg_rep_norm = psg_rep / psg_rep.norm(dim=1)[:, None]
            scores = torch.matmul(query_rep_norm, psg_rep_norm.transpose(0, 1)) * 20.0
            '''
            拿到了scores，下面开始求loss
            根据DPR的论文公式，我们可以用交叉熵来作为loss
            这里交叉熵的label或者说classes，就是我们的正样本。
            比如一个batch过来了8个q-p pairs，那么我们就有8个class，question i的positive是passage i，其他都是negative，用one-hot编码可以很方便地表示
            交叉熵如果要手动实现，那么就需要先求一个softmax，结果得到的是预测的概率值，然后取log，再求矩阵的对角线元素之和，最后取一个负值
            调用F.cross_entropy可以很快地计算交叉熵，target就是各个question的positive的下标
            PS：F.cross_entropy给的是avg loss，我们手动求出来的是一个batch的total loss
            '''
            # scores_soft = torch.softmax(scores, dim=1)
            # scores_soft_log = torch.log(scores_soft)
            # loss = -torch.sum(torch.diag(scores_soft_log))
            # print(loss / 4)
            target = torch.arange(query_rep_norm.size()[0], dtype=torch.long, device = device)
            loss = F.cross_entropy(scores, target)
            return loss # forward返回的结果就是模型输出的结果
        else:
            return {'encoded_text':pooled_seq_rep}


class Encoder:
    def __init__(self, model_path, batch_size=1000, pool_type='cls', max_seq_length=128):
        self.model_path = model_path
        self.pool_type = pool_type
        self.max_seq_length = max_seq_length
        self.device = torch.device("cpu") if not torch.cuda.is_available else torch.device("cuda:0")
        self.batch_size = batch_size
        self.model = BertForMatching.from_pretrained(model_path, pool_type=pool_type, is_train=False).eval().to(
            self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True, use_fast=True)


    def predict(self, inputs):
        '''

        :param inputs: 自然语言，就是我们说的话
        :return: 经过模型处理后的高维度向量
        '''
        encoded_inputs = self.tokenizer.batch_encode_plus(
            inputs,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )

        for k, v in encoded_inputs.items():
            encoded_inputs[k] = v.to(self.device)
        output = self.model(**encoded_inputs)
        for k, v in output.items():
            output[k] = v.detach().cpu().numpy()
        return output



    def encode_fn(self, examples): # 批处理map用的函数
        batch_texts = examples['content']
        encoded_batch_texts = self.predict(batch_texts) # 经过模型处理的corpus
        return encoded_batch_texts



    def encode_corpus(self, in_file, out_file):
        corpus = load_dataset('csv', data_files=in_file, delimiter='\t')['train']
        print('------ corpus编码开始 ------')
        encoded_corpus = corpus.map(self.encode_fn, batched=True, batch_size=self.batch_size) # 默认的batchsize是1000，会爆显存

        encoded_corpus.to_json(out_file)


class Searcher: # 检索器，输入query，输出topk相关的passage
    def __init__(self, encoder, index_file_path, topk=100, nlist=1000, nprobe=300):
        self.index_file = load_dataset('json', data_files=index_file_path)['train']
        self.encoder = encoder
        self.topk = topk
        self.nlist = nlist
        self.nprobe=nprobe
        self.build_index()

    def build_index(self):
        reps = np.array(self.index_file['encoded_text'], dtype=np.float32)
        quantizer = faiss.IndexFlatL2(reps.shape[1])
        self.index = faiss.IndexIVFFlat(quantizer, reps.shape[1], self.nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.train(reps)
        self.index.nprobe = self.nprobe
        self.index.add(reps)
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)


    def search(self, query):
        q_rep = self.encoder.predict([query])['encoded_text']
        scores, indices = self.index.search(q_rep, self.topk)
        return scores[0], [self.index_file['pid'][i] for i in indices[0]] # 因为我们每次只输入一个query，所以要取0




