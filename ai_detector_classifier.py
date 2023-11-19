#encoding=utf-8

import os
max_length = 256
train_data_path = "./data/train_essays.csv"
train_prompts_path = "./data/train_prompts.csv"
supplement_data_dir = "./data/archive/"
supplement_data_files = [  os.path.join(supplement_data_dir,f)  
                          for f in os.listdir(supplement_data_dir)
                          if(f.endswith('.csv') and '04' in f)]

def set_proxy():
    import os
    cache_dir = "/home/tx/workspace/cache"  # 替换为你想要的缓存目录的路径
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    # 代理
    os.environ['http_proxy'] = 'http://127.0.0.1:7890'
    os.environ['https_proxy'] = 'http://127.0.0.1:7890'
    os.environ['no_proxy'] = '127.0.0.1,localhost'
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['NO_PROXY'] = '127.0.0.1,localhost'
    return
set_proxy()
#############################################################
import pandas as pd
train_data=pd.read_csv(train_data_path)
train_prompts = pd.read_csv(train_prompts_path)
instructions = {
    0:train_prompts['instructions'][0],
    1:train_prompts['instructions'][1],
}

train_data['prompt'] = train_data.apply(
    lambda r: instructions[r['prompt_id']] if  r['prompt_id'] in instructions else -1,axis=1
)

supplement_train_data_v1 = pd.read_csv(supplement_data_dir+"train_essays_RDizzl3_seven_v1.csv")
supplement_train_data_v1['generated'] =  supplement_train_data_v1['label']
supplement_train_data_v1['prompt']= None
# train_data_re = pd.DataFrame(train_data[['text','generated','prompt']])

supplement_train_data = pd.concat([
    pd.read_csv(f)
    for f in supplement_data_files
])
# 生成数据只取200
supplement_train_data = supplement_train_data[supplement_train_data["label"]==1]
supplement_train_data['generated'] = supplement_train_data['label']
supplement_train_data = supplement_train_data[['text','generated','prompt']]
print(supplement_train_data.generated.value_counts())
print(supplement_train_data.sample(10))

#



import gc
#del train_data_re
gc.collect()
train_data_re = pd.DataFrame(train_data[['text','generated','prompt']])
train_data_all = pd.concat([train_data,supplement_train_data,supplement_train_data_v1])
print(train_data_all.generated.value_counts())
print(train_data_all.sample(10))

#############################################################################
cache_dir = "/home/tx/workspace/cache"  # 替换为你想要的缓存目录的路径
from transformers import AutoModelForSequenceClassification,AutoTokenizer
# from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
# class LlavaConfig(LlamaConfig):
#     model_type = "llava"
# AutoConfig.register("llava", LlavaConfig)
model_name = "bigscience/bloom-3b"
#model_name = "ChocoWu/nextgpt_7b_tiva_v0"#"liuhaotian/llava-v1.5-7b"
original_model = AutoModelForSequenceClassification.from_pretrained(model_name,cache_dir=cache_dir,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
# tokenizer.add_tokens(["[PAD]"])
print(original_model)

###########################################################################
def generate_prompt_ids(feature_text,tokenizer,max_length):

    feature_text_ds = tokenizer.encode(feature_text)


     
    if(len(feature_text_ds)  > max_length):
        
        ellipsis = tokenizer.encode('...')
        feature_len = max_length  - len(ellipsis)
        head_part_feature = feature_text_ds[ : int(feature_len / 2)]
        tail_part_feature = feature_text_ds[ (0 - int(feature_len / 2)): ]

        input_ids = head_part_feature + ellipsis + tail_part_feature 
    else:
        input_ids = feature_text_ds
    final_text = tokenizer.decode(input_ids)
    return input_ids,final_text

from torch.utils.data import TensorDataset,DataLoader,RandomSampler
import torch
from tqdm.auto import tqdm

class myDataset(TensorDataset):

    def __init__(self, datalist,max_length=256,tokenizer=None) -> None:
        super(myDataset,self).__init__()

        if(isinstance(datalist,pd.DataFrame)):
            self.datalist = datalist.to_dict(orient='list')
        elif(isinstance(datalist,dict)):
            self.datalist = datalist
        else:
            raise Exception("错误输入类型")
        self.tokenizer = tokenizer
        self.max_length = max_length
    

    def __len__(self):
        return len(self.datalist['text'])
    
    def __getitem__(self, index):
        text = self.datalist['text'][index]   

        labels = None
        label_length = 0
        if('generated' in self.datalist):
            labels = self.datalist['generated'][index] if self.datalist['generated'][index] is not None else 0
            pred_text = tokenizer.bos_token + text + tokenizer.eos_token
            input_ids,final_text = generate_prompt_ids(pred_text,self.tokenizer,max_length=self.max_length)#self.tokenizer.encode(final_text)
            att_mask = [1] * len(input_ids)
        else:
            input_ids = generate_prompt_ids(text,self.tokenizer,max_length=self.max_length-label_length)#self.tokenizer.encode(final_text)
            att_mask = [1] * len(input_ids)
            


        while(len(input_ids) < self.max_length):
            input_ids.append(self.tokenizer.pad_token_id)
            att_mask.append(0)
        input_ids = input_ids[:self.max_length]
        att_mask = att_mask[:self.max_length]
        print(len(input_ids),labels)

        if(labels is not None):
            return {'input_ids':torch.LongTensor(input_ids),'labels':torch.LongTensor(labels),'att_mask':torch.LongTensor(att_mask)},labels,final_text
        else:
            return {'input_ids':torch.LongTensor(input_ids),'att_mask':torch.LongTensor(att_mask)}

train_data_all_sample = train_data_all#train_data_all.groupby(['generated']).sample(n=4000,replace=False)
dataset = myDataset(datalist=train_data_all_sample,max_length=max_length,tokenizer=tokenizer) 
#dataset.preprocess()

dataloader_random = DataLoader(dataset, batch_size=6, sampler=RandomSampler(dataset))
#####################################################################################################
from peft import (
    LoraConfig,
    get_peft_model,
)

from peft import LoraConfig, TaskType
#lora_target_modules = ["query_key_value"]
lora_target_modules = [ f"transformer.h.{ly}.self_attention.query_key_value" for ly in range(25,29) ] \
                + ['score']
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                         inference_mode=False, target_modules=lora_target_modules,
                         r=4, lora_alpha=16, lora_dropout=0.1)
model = get_peft_model(original_model,peft_config)
print(model.print_trainable_parameters())



import torch 
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

#####################################################################################################
import torch
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
model = model.cuda()
optimizer = torch.optim.AdamW(
    [
        {'params': [p for p in model.parameters() if p.requires_grad],'lr': 5e-5},
    ]
)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(dataloader_random) ),
)


####################################################################################
all_path=[]
def save(model,step):
    if(len(all_path) > 0):
        last_path = all_path[-1] 
        idx = int(last_path.split('_')[-1])
        idx = idx + step
    else:
        idx = 0

    path = f"../../saved_model/ai_detector_peft_{idx}"
    model.save_pretrained(path)
    return



####################################################################################
def train():
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader_random)
        for step, batch_a in enumerate(pbar):
            batch,_,final_text = batch_a
            batch = {k: v.cuda() for k, v in batch.items()}
            #print(batch)
            #outputs = model(batch['input_ids'],labels=batch['labels'])
            labels_tensor = batch['labels']
            outputs = model(batch['input_ids'])
            
            logits = outputs.logits

            #loss = outputs.loss
            loss = loss_fn(torch.nn.softmax(logits,dim=-1).view(-1,2), labels_tensor.view(-1))
        
            total_loss += loss.detach().float()
            #print(loss)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if(step % 100 == 0):
                pbar.set_description(f"step {step} loss {loss}")
            if(step % 1000 == 0):
                save(model, 1000)
train()           
##########################################################################


model.save_pretrained("../../saved_model/ai_detector_peft_final")


#model = model.merge_and_unload()
#import torch
#torch.save(model.state_dict(),"../../saved_model/ai_detector.bin")


