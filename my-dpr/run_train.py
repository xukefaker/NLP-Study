from transformers import Trainer, DataCollatorWithPadding, AutoTokenizer, BertModel, AutoConfig, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from utils import CollatorWithPaddingTruncation, BertForMatching
import torch
from tqdm import tqdm


checkpoint_name = 'bert-base-multilingual-cased'
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
epoch_num = 6

dataset = load_dataset('json', data_files='../data/medical/train.json')['train']

tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
data_collator = CollatorWithPaddingTruncation(tokenizer, 128)
train_dataloader = DataLoader(
    dataset, shuffle=True, batch_size=32, collate_fn=data_collator
) # dataloader通过下标就可以访问一个个batch的数据


model = BertForMatching.from_pretrained(checkpoint_name)
model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=5e-5)


for epoch in range(epoch_num):
    count = 0
    print('-----epoch {}------'.format(epoch+1))
    for batch in tqdm(train_dataloader):
        count += 1
        batch = {k: v.to(device) for k, v in batch.items()}  # 记得转一下tensor的device
        loss = model(**batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if count % 200 == 0:
            print('loss {}'.format(loss))
    model.save_pretrained(save_directory='checkpoints/{}_epch-{}'.format(checkpoint_name, epoch))
tokenizer.save_pretrained('checkpoints/{}_epch-{}'.format(checkpoint_name, epoch))












