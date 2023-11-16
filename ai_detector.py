#encoding=utf-8

import os

train_data_path = "./data/train_essays.csv"
train_prompts_path = "./data/train_prompts.csv"
supplement_data_dir = "./data/archive/"
supplement_data_files = [  os.path.join(supplement_data_dir,f)  
                          for f in os.listdir(supplement_data_dir)
                          if(f.endswith('.csv'))]

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

supplement_train_data = pd.concat([
    pd.read_csv(f)
    for f in supplement_data_files
])
supplement_train_data['generated'] = supplement_train_data['label']
supplement_train_data = supplement_train_data[['text','generated','prompt']]
import gc
#del train_data_re
gc.collect()
train_data_re = pd.DataFrame(train_data[['text','generated','prompt']])
train_data_all = pd.concat([train_data,supplement_train_data])
#############################################################################
cache_dir = "/home/tx/workspace/cache"  # 替换为你想要的缓存目录的路径
from transformers import AutoModelForCausalLM,AutoTokenizer
# from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
# class LlavaConfig(LlamaConfig):
#     model_type = "llava"
# AutoConfig.register("llava", LlavaConfig)
model_name = "bigscience/bloom-3b"
#model_name = "ChocoWu/nextgpt_7b_tiva_v0"#"liuhaotian/llava-v1.5-7b"
original_model = AutoModelForCausalLM.from_pretrained(model_name,cache_dir=cache_dir,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
# tokenizer.add_tokens(["[PAD]"])

###########################################################################
def generate_prompt(prompt,feature_text):
    """
    
    """
    text=  f"""
        read the following text, understand its style and words choices:
        {feature_text}.
        is this text generated by AI?
    """
    return text

from torch.utils.data import TensorDataset,DataLoader,RandomSampler
import torch
from tqdm.auto import tqdm

class myDataset(TensorDataset):

    def __init__(self, datalist,max_length=256,tokenizer=None,preprocess_func= None) -> None:
        super(myDataset,self).__init__()

        if(isinstance(datalist,pd.DataFrame)):
            self.datalist = datalist.to_dict(orient='list')
        elif(isinstance(datalist,dict)):
            self.datalist = datalist
        else:
            raise Exception("错误输入类型")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess_func = preprocess_func
    
    def preprocess(self):
        datalist_tmp = {
            "text":[],
            "prompt":[],
            "generated":[]
        }
        for idx in tqdm(range(len(self))):
           _, act_len,_,_ = self[idx]
           if(act_len > self.max_length):
               continue
           
           datalist_tmp['text'].append(self.datalist['text'][idx])
           datalist_tmp['prompt'].append(self.datalist['prompt'][idx])
           datalist_tmp['generated'].append(self.datalist['generated'][idx])
        
        self.datalist = datalist_tmp
        return



    def __len__(self):
        return len(self.datalist['text'])
    
    def __getitem__(self, index):
        text = self.datalist['text'][index]  
        prompt = self.datalist['prompt'][index]  

        final_text = self.preprocess_func(prompt,text)
        input_ids = self.tokenizer.encode(final_text)
        att_mask = [1] * len(input_ids)


        labels = None
        if('generated' in self.datalist):
            generated = self.datalist['generated'][index]
            label_text = "yes, the text is generated" if generated > 0 else "no, the text is written by students"
            label_ids = self.tokenizer.encode(label_text)

            #final_text = [final_text,label_text]

            labels = [self.tokenizer.pad_token_id]  * len(input_ids)
            labels = labels + label_ids + [self.tokenizer.eos_token_id]
            input_ids = input_ids + label_ids   
            att_mask = [1] * len(input_ids)
        act_len = len(input_ids)
        while(len(input_ids) < self.max_length):
            input_ids.append(self.tokenizer.pad_token_id)
            if(labels is not None):
                labels.append(self.tokenizer.pad_token_id)
            att_mask.append(0)
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        att_mask = att_mask[:self.max_length]
        if(labels is not None):
            return {'input_ids':torch.LongTensor(input_ids),'labels':torch.LongTensor(labels),'att_mask':torch.LongTensor(att_mask)},act_len,final_text,label_text
        else:
            return {'input_ids':torch.LongTensor(input_ids),'att_mask':torch.LongTensor(att_mask)}

train_data_all_sample = train_data_all.groupby(['generated']).sample(n=4000,replace=False)
dataset = myDataset(datalist=train_data_all_sample,max_length=512,tokenizer=tokenizer,preprocess_func=generate_prompt) 
dataset.preprocess()

dataloader_random = DataLoader(dataset, batch_size=2, sampler=RandomSampler(dataset))
#####################################################################################################
from peft import (
    LoraConfig,
    get_peft_model,
)

from peft import LoraConfig, TaskType
#lora_target_modules = ["query_key_value"]
lora_target_modules = [ f"transformer.h.{ly}.self_attention.query_key_value" for ly in range(25,29) ]

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                         inference_mode=False, target_modules=lora_target_modules,
                         r=4, lora_alpha=16, lora_dropout=0.1)
model = get_peft_model(original_model,peft_config)
print(model.print_trainable_parameters())



import torch 
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=tokenizer.pad_token_id)
def generate(prompt,model_,tokenizer_):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(model_.device)
    #model_.disable_adapter()
    # 生成文本
    inputs = {"input_ids":input_ids, "max_length":512, 
              "num_beams":5, "no_repeat_ngram_size":2,
             "top_k":50, "top_p":0.95}
    with torch.no_grad():
        output = model_.generate(**inputs)
    # 将生成的token解码成文本
    generated_text = tokenizer_.decode(output[0], skip_special_tokens=True)
    return generated_text

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

def train():
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader_random)
        for step, batch_a in enumerate(pbar):
            batch,_,final_text,label_text = batch_a
            batch = {k: v.cuda() for k, v in batch.items()}
            #print(batch)
            #outputs = model(batch['input_ids'],labels=batch['labels'])
            labels_tensor = batch['labels']
            outputs = model(batch['input_ids'])
            
            logits = outputs.logits
            logits = logits[...,:-1,:].contiguous()
            labels_tensor = labels_tensor[...,1:].contiguous()
            
            
            #loss = outputs.loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels_tensor.view(-1))
        
            total_loss += loss.detach().float()
            #print(loss)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if(step % 100 == 0):
                pbar.set_description(f"step {step} loss {loss}")
                print(loss)
            if(step  % 200 == 0 and step > 0):
                test = generate(final_text[0],model,tokenizer)
                print("="* 100)
                print(">"* 100)
                print(test)
                print(">"* 10,label_text[0])
train()           
##########################################################################

model.save_pretrained("../../saved_model/ai_detector_peft")
#model = model.merge_and_unload()
#import torch
#torch.save(model.state_dict(),"../../saved_model/ai_detector.bin")


