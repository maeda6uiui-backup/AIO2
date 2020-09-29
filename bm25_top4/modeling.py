import argparse
import logging
import os
import json
import sys
import random
import torch
import torch.nn as nn
import numpy as np
import math
import MeCab
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader,TensorDataset
from transformers import BertForMultipleChoice,AdamW,get_linear_schedule_with_warmup

import hashing

#Fix the seed.
SEED=42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

#Set up a logger.
logging_fmt = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_TOTAL_DOCS=114229
AVGDL=1961

class InputExample(object):
    def __init__(self, qid, question, endings, label=None):
        self.qid = qid
        self.question = question
        self.endings = endings
        self.label = label

def load_examples(example_filepath):
    with open(example_filepath, "r", encoding="utf_8") as r:
        lines = r.read().splitlines()

    examples = []
    for line in lines:
        data = json.loads(line)

        qid = data["qid"]
        question = data["question"].replace("_", "")
        options = data["answer_candidates"]
        answer = data["answer_entity"]

        label=0
        if answer!="":
            label=options.index(answer)

        example = InputExample(qid, question, options, label)
        examples.append(example)

    return examples

def create_dataset(input_dir,num_examples=-1,num_options=4):
    input_ids=torch.load(os.path.join(input_dir,"input_ids.pt"))
    attention_mask=torch.load(os.path.join(input_dir,"attention_mask.pt"))
    token_type_ids=torch.load(os.path.join(input_dir,"token_type_ids.pt"))
    labels=torch.load(os.path.join(input_dir,"labels.pt"))

    input_ids=input_ids[:,:num_options,:]
    attention_mask=attention_mask[:,:num_options,:]
    token_type_ids=token_type_ids[:,:num_options,:]

    indices=torch.empty(input_ids.size(0),dtype=torch.long).to(device)
    for i in range(input_ids.size(0)):
        indices[i]=i

    if num_examples>0:
        input_ids=input_ids[:num_examples,:,:]
        attention_mask=attention_mask[:num_examples,:,:]
        token_type_ids=token_type_ids[:num_examples,:,:]
        labels=labels[:num_examples]
        indices=indices[:num_examples]

    return TensorDataset(input_ids,attention_mask,token_type_ids,labels,indices)

def get_d_and_frequency(count_filepath,qi):
    with open(count_filepath,"r",encoding="utf_8") as r:
        lines=r.read().splitlines()

    d=int(lines[0])

    frequency=0
    for i in range(1,len(lines)):
        splits=lines[i].split("\t")
        word=splits[0]
        if word==qi:
            frequency=int(splits[1])
            break
    
    return d,frequency

def calc_score(mecab,question,option,count_dir,nqis,ignores,k1=1.6,b=0.75,delta=1.0):
    genkeis=[]
    node=mecab.parseToNode(question)
    while node:
        features=node.feature.split(",")

        hinsi=features[0]
        if hinsi=="BOS/EOS":
            node=node.next
            continue

        genkei=features[6]
        genkeis.append(genkei)

        node=node.next

    score=0
    for genkei in genkeis:
        idf=0
        #if genkei not in ignores:
        nqi=0
        if genkei in nqis:
            nqi=nqis[genkei]

        idf=math.log((NUM_TOTAL_DOCS-nqi+0.5)/(nqi+0.5))
        idf=max(0,idf)

        option_hash=hashing.get_md5_hash(option)
        count_filepath=os.path.join(count_dir,option_hash+".txt")
        d,freq=get_d_and_frequency(count_filepath,genkei)

        numerator=freq*(k1+1)
        denominator=freq+k1*(1-b+b*d/AVGDL)

        score+=idf*(numerator/denominator+delta)

    return score

def create_train_dataset(mecab,examples,count_dir,nqis,ignores,dataset,dataset_save_dir):
    num_examples=len(dataset)

    input_ids=torch.empty(num_examples,4,512,dtype=torch.long).to(device)
    attention_mask=torch.empty(num_examples,4,512,dtype=torch.long).to(device)
    token_type_ids=torch.empty(num_examples,4,512,dtype=torch.long).to(device)
    labels=torch.empty(num_examples,dtype=torch.long).to(device)

    for i,data in tqdm(enumerate(dataset),total=len(dataset)):
        #0番目のデータは必ず含む。
        input_ids[i,0]=data[0][0]
        attention_mask[i,0]=data[1][0]
        token_type_ids[i,0]=data[2][0]
        labels[i]=data[3]

        example_index=data[4]
        example=examples[example_index]

        scores=np.empty(20)
        for j in range(20):
            score=calc_score(mecab,example.question,example.endings[j],count_dir,nqis,ignores)
            scores[j]=score

        top_4_indices=(-scores).argsort()[:4]
        top_3_indices=np.empty(3,dtype=np.int64)

        already_set_count=0
        for j in range(4):
            if top_4_indices[j]!=0:
                top_3_indices[already_set_count]=top_4_indices[j]
                already_set_count+=1
            if already_set_count==3:
                break

        for j in range(3):
            input_ids[i,j+1]=data[0][top_3_indices[j]]
            attention_mask[i,j+1]=data[1][top_3_indices[j]]
            token_type_ids[i,j+1]=data[2][top_3_indices[j]]

    #Save the train dataset.
    logger.info("Save train dataset to {}.".format(dataset_save_dir))
    os.makedirs(dataset_save_dir,exist_ok=True)

    input_ids_filepath=os.path.join(dataset_save_dir,"input_ids.pt")
    attention_mask_filepath=os.path.join(dataset_save_dir,"attention_mask.pt")
    token_type_ids_filepath=os.path.join(dataset_save_dir,"token_type_ids.pt")
    labels_filepath=os.path.join(dataset_save_dir,"labels.pt")
    torch.save(input_ids,input_ids_filepath)
    torch.save(attention_mask,attention_mask_filepath)
    torch.save(token_type_ids,token_type_ids_filepath)
    torch.save(labels,labels_filepath)

    return TensorDataset(input_ids,attention_mask,token_type_ids,labels)

def train(classifier_model,optimizer,scheduler,dataloader):
    classifier_model.train()

    count_steps=0
    total_loss=0

    for batch_idx,batch in enumerate(dataloader):
        batch = tuple(t for t in batch)
        bert_inputs = {
            "input_ids": batch[0].to(device),
            "attention_mask": batch[1].to(device),
            "token_type_ids": batch[2].to(device),
            "labels": batch[3].to(device)
        }

        # Initialize gradiants
        optimizer.zero_grad()
        # Forward propagation
        classifier_outputs=classifier_model(**bert_inputs)
        loss=classifier_outputs[0]
        # Backward propagation
        loss.backward()
        # Update parameters
        optimizer.step()
        scheduler.step()

        count_steps+=1
        total_loss+=loss.item()

        if batch_idx%100==0:
            logger.info("Step: {}\tLoss: {}\tlr: {}".format(
                batch_idx,loss.item(),optimizer.param_groups[0]["lr"]))

    return total_loss/count_steps

def simple_accuracy(pred_labels, correct_labels):
    return (pred_labels == correct_labels).mean()

def evaluate(classifier_model,dataloader,examples,mecab,count_dir,nqis,ignores):
    classifier_model.eval()

    preds = None
    correct_labels = None
    for batch_idx,batch in tqdm(enumerate(dataloader),total=len(dataloader)):
        with torch.no_grad():
            batch = tuple(t for t in batch)
            batch_size=batch[0].size(0)
            bert_inputs = {
                "input_ids": batch[0].to(device),
                "attention_mask": batch[1].to(device),
                "token_type_ids": batch[2].to(device),
                "labels": batch[3].to(device)
            }

            classifier_outputs=classifier_model(**bert_inputs)
            logits=classifier_outputs[1]
            logits=logits.detach().cpu().numpy()

            sorted_logits=np.sort(logits,axis=1)
            example_indices=batch[4].to(device)
            for i in range(batch_size):
                example_index=example_indices[i]
                example=examples[example_index]

                scores=np.empty(20)
                for j in range(20):
                    score=calc_score(mecab,example.question,example.endings[j],count_dir,nqis,ignores)
                    scores[j]=score
                
                logits[i]*=scores

            if preds is None:
                preds = logits
                correct_labels = bert_inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits, axis=0)
                correct_labels = np.append(
                    correct_labels, bert_inputs["labels"].detach().cpu().numpy(), axis=0
                )

    pred_labels = np.argmax(preds, axis=1)
    accuracy = simple_accuracy(pred_labels, correct_labels)

    return pred_labels,correct_labels,accuracy

def main(
    batch_size,num_epochs,
    lr,train_input_dir,dev1_input_dir,
    train_example_filepath,dev1_example_filepath,
    count_dir,nqis_filepath,ignores_filepath,
    train_dataset_save_dir,result_save_dir):
    logger.info("seed: {}".format(SEED))
    logger.info("batch_size: {} num_epochs: {} lr: {}".format(batch_size,num_epochs,lr))

    logger.info("Create dev1 dataloader from {}.".format(dev1_input_dir))
    dev1_dataset=create_dataset(dev1_input_dir,num_examples=-1,num_options=20)
    dev1_dataloader=DataLoader(dev1_dataset,batch_size=4,shuffle=False,drop_last=True)

    #Create a classifier model.
    logger.info("Create a classifier model.")
    classifier_model=BertForMultipleChoice.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    classifier_model.to(device)

    #Load nqis.
    nqis={}
    with open(nqis_filepath,"r",encoding="utf_8") as r:
        lines=r.read().splitlines()
    for line in lines:
        splits=line.split("\t")
        word=splits[0]
        count=int(splits[1])

        nqis[word]=count

    #Load ignores.
    ignores=[]
    with open(ignores_filepath,"r",encoding="utf_8") as r:
        lines=r.read().splitlines()
    for line in lines:
        ignores.append(line)

    mecab=MeCab.Tagger()

    #Load examples.
    logger.info("Loading examples.")
    train_examples=load_examples(train_example_filepath)
    dev1_examples=load_examples(dev1_example_filepath)

    logger.info("Start creating train dataloader.")
    if os.path.exists(train_dataset_save_dir):
        logger.info("Load data from {}.".format(train_dataset_save_dir))

        input_ids_filepath=os.path.join(train_dataset_save_dir,"input_ids.pt")
        attention_mask_filepath=os.path.join(train_dataset_save_dir,"attention_mask.pt")
        token_type_ids_filepath=os.path.join(train_dataset_save_dir,"token_type_ids.pt")
        labels_filepath=os.path.join(train_dataset_save_dir,"labels.pt")

        input_ids=torch.load(input_ids_filepath,map_location=device)
        attention_mask=torch.load(attention_mask_filepath,map_location=device)
        token_type_ids=torch.load(token_type_ids_filepath,map_location=device)
        labels=torch.load(labels_filepath,map_location=device)

        train_dataset=TensorDataset(input_ids,attention_mask,token_type_ids,labels)
    else:
        logger.info("Create train dataset from {}.".format(train_input_dir))

        train_dataset=create_dataset(train_input_dir,num_examples=-1,num_options=20)
        train_dataset=create_train_dataset(
            mecab,train_examples,
            count_dir,nqis,ignores,
            train_dataset,train_dataset_save_dir
        )

    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

    #Create a directory to save the results in.
    os.makedirs(result_save_dir,exist_ok=True)

    #Create an optimizer and a scheduler.
    optimizer=AdamW(classifier_model.parameters(),lr=lr,eps=1e-8)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    logger.info("Start model training.")
    for epoch in range(num_epochs):
        logger.info("===== Epoch {}/{} =====".format(epoch+1,num_epochs))

        mean_loss=train(classifier_model,optimizer,scheduler,train_dataloader)
        logger.info("Mean loss: {}".format(mean_loss))

        #Save model parameters.
        checkpoint_filepath=os.path.join(result_save_dir,"checkpoint_{}.pt".format(epoch+1))
        torch.save(classifier_model.state_dict(),checkpoint_filepath)

        pred_labels,correct_labels,accuracy=evaluate(
            classifier_model,dev1_dataloader,
            dev1_examples,mecab,count_dir,nqis,ignores)
        logger.info("Accuracy: {}".format(accuracy))

        #Save results as text files.
        res_filepath=os.path.join(result_save_dir,"result_eval_{}.txt".format(epoch+1))
        labels_filepath=os.path.join(result_save_dir,"labels_eval_{}.txt".format(epoch+1))

        with open(res_filepath,"w") as w:
            w.write("Accuracy: {}\n".format(accuracy))

        with open(labels_filepath,"w") as w:
            for pred_label,correct_label in zip(pred_labels,correct_labels):
                w.write("{} {}\n".format(pred_label,correct_label))

    logger.info("Finished model training.")

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--batch_size",type=int,default=4)
    parser.add_argument("--num_epochs",type=int,default=20)
    parser.add_argument("--lr",type=float,default=5e-5)
    parser.add_argument("--train_input_dir",type=str,default="~/EncodedTextTohoku/Train")
    parser.add_argument("--dev1_input_dir",type=str,default="~/EncodedTextTohoku/Dev1")
    parser.add_argument("--train_example_filepath",type=str)
    parser.add_argument("--dev1_example_filepath",type=str)
    parser.add_argument("--count_dir",type=str)
    parser.add_argument("--nqis_filepath",type=str)
    parser.add_argument("--ignores_filepath",type=str)
    parser.add_argument("--train_dataset_save_dir",type=str,default="./OutputDir")
    parser.add_argument("--result_save_dir",type=str,default="./OutputDir")

    args=parser.parse_args()

    main(
        args.batch_size,
        args.num_epochs,
        args.lr,
        args.train_input_dir,
        args.dev1_input_dir,
        args.train_example_filepath,
        args.dev1_example_filepath,
        args.count_dir,
        args.nqis_filepath,
        args.ignores_filepath,
        args.train_dataset_save_dir,
        args.result_save_dir
    )
