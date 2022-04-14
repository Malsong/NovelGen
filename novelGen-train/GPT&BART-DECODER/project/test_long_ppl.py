import transformers
from transformers import BertTokenizer
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import os
import re
import random
import argparse
import numpy as np
from datetime import datetime
from torch.nn import DataParallel
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from numpy import mean
import pandas as pd

# 自定义dataset(不区分训练、验证、测试的)
class MyDataset(Dataset):
  def __init__(self, tokenized_data_path, num_pieces_list, shift_tag, stride = 1024, seq_length = 1024, transform = None, target_transform = None):
    self.tokenized_data_path = tokenized_data_path  # e.g. "/content/drive/MyDrive/Colab/data/"
    self.seq_length = seq_length
    self.stride = stride

    samples = []
    for i in num_pieces_list:
      with open(self.tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
        line = f.read().strip()
      tokens = line.split()
      tokens = [int(token) for token in tokens]
      start_point = 0
      # 每个seq_length取一次语料作为数据
      while start_point < len(tokens) - self.seq_length:
        samples.append(tokens[start_point: start_point + self.seq_length])
        start_point += self.stride
      if start_point < len(tokens):
        samples.append(tokens[len(tokens)-self.seq_length:])

    random.shuffle(samples)
    data = []
    for ids in samples:
      ids_for_input = [int(x) for x in ids]
      ids_for_label = [int(x) for x in ids]

      if shift_tag == 0 :  # 无需手动移位
        data.append(( ids_for_input, ids_for_label))
      elif shift_tag == 1 : # 需要手动移位
        data.append(( ids_for_input[:-1], ids_for_label[1:]))

    random.shuffle(data)
    data = torch.tensor(data)
    self.data = data
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self,index):
    inp_one,tag_one = self.data[index]
    return inp_one,tag_one           #这就返回一个样本
 
  def __len__(self):
	  return len(self.data) 

# 创建test_dataloader
def getDatasetAlreadySplit(tokenized_data_path,num_pieces,shift_tag,stride,seq_length,batch_size,test_num_pieces):
  test_datasets = MyDataset(tokenized_data_path,test_num_pieces,shift_tag,stride,seq_length)
  test_dataloader = torch.utils.data.DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=False)
  return test_dataloader 

# 评估ppl 
def eval(dataloader,model,device,n_ctx,model_choice):
    model.eval()
    PPL_List = []
    size = len(dataloader)
    total_loss = 0
    with torch.no_grad():
        for i,(batch_inputs,batch_labels) in enumerate(dataloader):
            #  prepare data
            batch_inputs = batch_inputs.long().to(device)
            batch_labels = batch_labels.long().to(device)
            #print('batch_inputs:',batch_inputs)
            #print('batch_labels:',batch_labels)
            bs, sl = batch_inputs.size()
            outputs = model(batch_inputs, labels=batch_labels)
            loss, logits = outputs[:2]
            if model_choice == 1 :
              shift_logits = logits[:, :-1, :].contiguous()
              shift_labels = batch_labels[:, 1:].contiguous()
            elif model_choice == 2 :
              shift_logits = logits.contiguous()
              shift_labels = batch_labels.contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=0, reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
            # print('loss: ',loss)  # 每个字的交叉熵损失列表
            meanloss = loss.sum(1) / n_ctx
            total_loss += sum(meanloss.cpu().numpy().tolist())
            # print('meanloss:',meanloss)
            ppl = torch.exp(meanloss).cpu().numpy().tolist()
            PPL_List.extend(ppl)
            # print('PPL_List: ',PPL_List)
            count = 0
            if (i+1) % 10 == 0:
                now_list = PPL_List[count*bs*10 : ]
                now_ppl = mean(now_list)
                print('lastest 10 batches ppl for {}/{}  = {}'.format((i+1),size,now_ppl))
                count +=1

    avg_ppl = mean(PPL_List)

    return PPL_List,avg_ppl,total_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='/content/drive/MyDrive/test-gpt/gpt2-common/config.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--tokenized_data_path', default='/content/drive/MyDrive/test-gpt/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--tokenizer_path', default='/content/drive/MyDrive/test-gpt/gpt2-common/vocab.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='batch size')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--model_choice', default=1, type=int, required=False, help='选择使用gpt2还是bart')
    parser.add_argument('--save_ppl_dir', default='', type=str, required=False, help='存储ppl路径')
    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    tokenized_data_path = args.tokenized_data_path
    tokenizer_path = args.tokenizer_path
    batch_size = args.batch_size
    log_step = args.log_step
    stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    max_grad_norm = args.max_grad_norm
    num_pieces = args.num_pieces
    save_ppl_dir = args.save_ppl_dir

    model_choice = args.model_choice   
    
    if model_choice == 1:
        full_tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    elif model_choice == 2:
        full_tokenizer = BertTokenizer.from_pretrained("uer/bart-base-chinese-cluecorpussmall")
    full_tokenizer.max_len = 99999

    if model_choice == 1 : # 选1为GPT2
        shift_tag = 0
        n_ctx = 1024
        if not args.pretrained_model:
            model_config = transformers.GPT2Config.from_json_file(args.model_config)
            model = transformers.GPT2LMHeadModel(config=model_config)
            print('config:\n' + model_config.to_json_string())
        else:
            model = transformers.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    elif model_choice == 2 : # 选2为Bart
        shift_tag = 1
        n_ctx = 512
        model = transformers.BartForCausalLM.from_pretrained(args.pretrained_model)
        assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."

    test_num_pieces = np.load('/content/drive/MyDrive/RUN_CLM/data/test_num_np.npy').tolist()
    test_dataloader=getDatasetAlreadySplit(tokenized_data_path,num_pieces,shift_tag,stride,n_ctx,batch_size,test_num_pieces)
    
    model.to(device)
    multi_gpu = False
    print('calculating total test steps')
    total_test_steps = len(test_dataloader)
    print('total test steps = {}'.format(test_dataloader))

    print('starting testing')
    test_loss_list = []
    test_avg_ppl_list = []

    test_PPL_List,test_avg_ppl,test_total_loss = eval(test_dataloader,model,device,n_ctx,model_choice)
    test_avg_loss = test_total_loss/(len(test_dataloader)*batch_size)
    print('average test ppl is ',test_avg_ppl)
    print('average test loss is ',test_avg_loss)

    #字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'test_PPL_List': test_PPL_List})
    print(dataframe)
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(save_ppl_dir,sep=',')

if __name__ == '__main__':
    main()
