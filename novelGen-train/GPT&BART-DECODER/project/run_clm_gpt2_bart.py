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

# 切分原始语料文本
def build_files(raw_data_path, tokenized_data_path, full_tokenizer, num_pieces):
    # raw_data_path 为语料文本文件夹名
    txtfiles = os.listdir(raw_data_path) 
    lines=[]
    for file in txtfiles:
        position = raw_data_path + '/' + file
        with open(position,"r",encoding='utf-8') as f:
            print('reading lines...')
            Lines_one = f.readlines()
            for line in Lines_one:
                if line !='/n':
                    lines.append(line) #去除空行
            # lines.append(' [CLS] ')  # 每本书后加[CLS] 区分为下一本书，有必要吗？
    print('file read finish!')
    # lines为所有文本文件的段落列表形式       
    lines = [line.replace('\n', ' [SEP] ') for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
    single = ''.join(lines)
    len_single = len(single)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    for i in tqdm(range(num_pieces)):  # 每次取全部文本single的一部分存储为piece
        single_ids = full_tokenizer.convert_tokens_to_ids(
            full_tokenizer.tokenize(single[len_single // num_pieces * i: len_single // num_pieces * (i + 1)]))
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
            for id in single_ids[:-1]:
                f.write(str(id) + ' ')
            f.write(str(single_ids[-1]))
            f.write('\n')
    print('txt split to pieces finish!')

# 创建并分割数据集（里面的datasets是全部数据）
def getDatasetAndSplit(tokenized_data_path,num_pieces,shift_tag,stride,seq_length,batch_size):
  num_pieces_list = np.arange(num_pieces)
  datasets = MyDataset(tokenized_data_path,num_pieces_list,shift_tag,stride,seq_length) # # seq_length: gpt2：最长不能超过1024 bart：最长不能超过512
  # 创造数据分割比例来分割训练、验证、测试集，默认 8：1：1
  dataset_size = len(datasets)
  indices = list(range(dataset_size))
  train_size = int(0.98 * dataset_size)
  val_size = int(0.01 * dataset_size)
  test_size = dataset_size - train_size - val_size
  train_indices, val_indices, test_indices = indices[:train_size], indices[train_size:train_size+val_size] ,  indices[train_size+val_size:]
  
  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(val_indices)
  test_sampler = SubsetRandomSampler(test_indices)

  train_dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size,sampler=train_sampler)
  validation_dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size,sampler=valid_sampler)
  test_dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size,sampler=test_sampler)
  return train_dataloader,validation_dataloader,test_dataloader

def getDatasetAlreadySplit(tokenized_data_path,num_pieces,shift_tag,stride,seq_length,batch_size,train_num_pieces,valid_num_pieces,test_num_pieces):
  train_datasets = MyDataset(tokenized_data_path,train_num_pieces,shift_tag,stride,seq_length)
  valid_datasets = MyDataset(tokenized_data_path,valid_num_pieces,shift_tag,stride,seq_length)
  test_datasets = MyDataset(tokenized_data_path,test_num_pieces,shift_tag,stride,seq_length)
  train_dataloader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=False)
  validation_dataloader = torch.utils.data.DataLoader(dataset=valid_datasets, batch_size=batch_size, shuffle=False)
  test_dataloader = torch.utils.data.DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=False)
  return train_dataloader,validation_dataloader,test_dataloader 


# 文本分句
def cut_sent(para):
    pattern = ['([。！？\?])([^"’])','(\.{6})([^"’])','(\…{2})([^"’])','([。！？\?]["’])([^，。！？\?])']
    for i in pattern:
        para = re.sub(i, r"\1\n\2", para)
    para = para.rstrip()
    return para.split("\n")

# 将其中被错分的语句进行连接(主要是针对话语)
def cut_and_connect(para):
    sentence_before = []
    sentence_after = []
    for each_para in para:
        sentence_before.append(cut_sent(each_para))
    # 核心代码！（将被错分的语句进行连接）
    for each in sentence_before:
        listL = []
        sentence = ""
        FLAG = True # 非常关键！判断有'：“'的符号后面的语句是否继续拼接
        for i in each:
            if i.find('："') * i.find('"') >= 0 and FLAG:
                listL.append(i + sentence)
            else:
                FLAG = False
                sentence = sentence + i
                if i.find('"') > 0:
                    listL.append(sentence)
                    sentence = ""
                    FLAG = True
        sentence_after.extend(listL)
    return sentence_after

# 训练
def train(train_dataloader,model,optimizer,device,epoch,multi_gpu,gradient_accumulation,max_grad_norm,scheduler,log_step):
    model.train()
    running_loss = 0
    total_loss = 0
    size = len(train_dataloader)
    for i,(batch_inputs,batch_labels) in enumerate(train_dataloader):
        step = i
        #  prepare data
        batch_inputs = batch_inputs.long().to(device)
        batch_labels = batch_labels.long().to(device)
        #  print('batch_inputs:',batch_inputs)
        #  forward pass
        outputs = model.forward(input_ids=batch_inputs, labels=batch_labels)
        loss, logits = outputs[:2]
        #  get loss
        if multi_gpu:
            loss = loss.mean()
        if gradient_accumulation > 1:
            loss = loss / gradient_accumulation
        #  loss backward
        loss = loss.requires_grad_() #加
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        #  optimizer step
        if (step + 1) % gradient_accumulation == 0:
            running_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        if (step + 1) % log_step == 0:
            print('now time: {}:{}. Step {}/{} of epoch {}, loss {}'.format(
                datetime.now().hour,
                datetime.now().minute,
                (step + 1) // gradient_accumulation,
                size,
                epoch + 1,
                running_loss / log_step))
            total_loss += running_loss
            running_loss = 0
    return total_loss / size
        

# 验证  
def eval(validation_dataloader,model,device,n_ctx,model_choice):
    model.eval()
    PPL_List = []
    size = len(validation_dataloader)
    total_loss = 0
    with torch.no_grad():
        for i,(batch_inputs,batch_labels) in enumerate(validation_dataloader):
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

    valid_avg_ppl = mean(PPL_List)

    return PPL_List,valid_avg_ppl,total_loss

# 测试数据存储
def test_sentences(test_dataloader,tokenizer):
    test_list = []
    for i,(batch_inputs,batch_labels) in enumerate(test_dataloader):
        batch_inputs = batch_inputs
        test_list.extend(batch_inputs.numpy().tolist())
    text = []
    for i in range(len(test_list)):
        text_one = tokenizer.convert_ids_to_tokens(test_list[i])
        for i, item in enumerate(text_one):
            if item == '[MASK]':
                text_one[i] = ''
            elif item == '[CLS]':
                text_one[i] = ''
            elif item == '[SEP]':
                text_one[i] = ''
        text_one = ''.join(text_one).replace('##', '').strip() # 文本
        text.append(text_one)

    sentences = []
    for i in range(len(text)):
        senten = cut_and_connect([text[i]])
        sentences.extend( senten[1:-1] )
    random.shuffle(sentences)

    count = 0
    final_sentences = []
    for s in sentences :
        if len(s)>=20 :
            final_sentences.append(s)
            count +=1
        if count >=100 :
            break
    return final_sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='/content/drive/MyDrive/test-gpt/gpt2-common/config.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--raw_data_path', default='/content/drive/MyDrive/test-gpt/data/dataG', type=str, required=False, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='/content/drive/MyDrive/test-gpt/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--tokenizer_path', default='/content/drive/MyDrive/newTrain/gpt2-common/vocab.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw', action='store_true', help='是否先做tokenize')
    parser.add_argument('--epochs', default=10, type=int, required=False, help='训练循环')
    parser.add_argument('--save_epochs_num', default=5, type=int, required=False, help='训练时隔几轮次存储一次模型')
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--output_dir', default='/content/drive/MyDrive/test-gpt/project/model_save/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='uer/bart-base-chinese-cluecorpussmall', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--test_sentences_path', default='/content/drive/MyDrive/test-gpt/project/model_save/', type=str, required=False, help='测试集语句输出路径')
    parser.add_argument('--model_choice', default=1, type=int, required=False, help='选择使用gpt2还是bart')
    parser.add_argument('--test_random', default=1, type=int, required=False, help='是否测试集每次都不一样')
    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    raw_data_path = args.raw_data_path
    tokenized_data_path = args.tokenized_data_path
    tokenizer_path = args.tokenizer_path
    raw = args.raw  # 选择是否从零开始构建数据集
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    max_grad_norm = args.max_grad_norm
    num_pieces = args.num_pieces
    output_dir = args.output_dir
    save_epochs_num = args.save_epochs_num
    model_choice = args.model_choice
    test_sentences_path = args.test_sentences_path
    test_random = args.test_random
    
    
    
    if model_choice == 1:
        full_tokenizer = BertTokenizer(vocab_file=tokenizer_path)
    elif model_choice == 2:
        full_tokenizer = BertTokenizer(vocab_file=tokenizer_path)
    full_tokenizer.max_len = 999999
    

    if raw:
        print('building files')
        build_files(raw_data_path=raw_data_path, tokenized_data_path=tokenized_data_path, full_tokenizer=full_tokenizer,
                    num_pieces=num_pieces)
        print('files built')

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

    if test_random == 1 :
      train_dataloader,validation_dataloader,test_dataloader=getDatasetAndSplit(tokenized_data_path,num_pieces,shift_tag,stride,n_ctx,batch_size)
    elif test_random == 0 :
      train_num_pieces = np.load('/content/drive/MyDrive/RUN_CLM/data/train_num_np.npy').tolist()
      valid_num_pieces = np.load('/content/drive/MyDrive/RUN_CLM/data/valid_num_np.npy').tolist()
      test_num_pieces = np.load('/content/drive/MyDrive/RUN_CLM/data/test_num_np.npy').tolist()
      train_dataloader,validation_dataloader,test_dataloader=getDatasetAlreadySplit(tokenized_data_path,num_pieces,shift_tag,stride,n_ctx,batch_size,train_num_pieces,valid_num_pieces,test_num_pieces)

    # 测试集数据存储
    final_sentences_test =test_sentences(test_dataloader,full_tokenizer)
    testList_np = np.array(final_sentences_test)
    np.save(test_sentences_path,testList_np)
    print('test data save as {}'.format(test_sentences_path))

    model.to(device)
    multi_gpu = False
    full_len = 0
    print('calculating total steps')
    total_train_steps = len(train_dataloader)*epochs
    print('total train steps = {}'.format(total_train_steps))

    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,num_training_steps = total_train_steps)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model)
        multi_gpu = True
        
    print('starting training')
    best_valid_avg_ppl = 100000
    best_epoch = 0
    train_loss_list = []
    valid_loss_list = []
    valid_avg_ppl_list = []
    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))

        # 训练函数
        train_avg_loss = train(train_dataloader,model,optimizer,device,epoch,multi_gpu,gradient_accumulation,max_grad_norm,scheduler,log_step)
        train_loss_list.append(train_avg_loss)
        # 验证函数
        valid_PPL_List,valid_avg_ppl,valid_total_loss = eval(validation_dataloader,model,device,n_ctx,model_choice)
        valid_avg_loss = valid_total_loss/(len(validation_dataloader)*batch_size)
        print('average valid ppl for epoch{} is '.format(epoch+1),valid_avg_ppl)

        valid_loss_list.append(valid_avg_loss)
        valid_avg_ppl_list.append(valid_avg_ppl)

        # 验证集的ppl小于之前最好的PPL，则存储当前轮次的模型为最好效果的模型
        if valid_avg_ppl < best_valid_avg_ppl:
            best_epoch = epoch + 1
            best_valid_avg_ppl = valid_avg_ppl
            print('saving now best model for epoch {}'.format(epoch + 1))
            if not os.path.exists(output_dir + 'best_model_epoch'):
                os.mkdir(output_dir + 'best_model_epoch')
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir + 'best_model_epoch')

        # 每隔save_epochs_num轮存储一次模型
        if (epoch+1) % save_epochs_num == 0 :
            print('saving model for epoch {}'.format(epoch + 1))
            if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
                os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
        
        print('epoch {} finished'.format(epoch + 1))
        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    print('training finished')
    print('the best epoch is ',best_epoch)
    #字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'train_loss': train_loss_list,'valid_loss': valid_loss_list,'valid_ppl': valid_avg_ppl_list})
    print(dataframe)
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(r"/content/drive/MyDrive/Gtrain_epochs_list.csv",sep=',')


if __name__ == '__main__':
    main()