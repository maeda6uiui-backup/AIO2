import argparse
import logging
import os
import sys
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader,TensorDataset
from transformers import (
    BertModel,
    BertForMultipleChoice,
    AdamW,
    get_linear_schedule_with_warmup,
)

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

class Options(object):
    def __init__(self):
        self.options=[]

    def append(self,option):
        self.options.append(option)

    def get(self,index):
        return self.options[index]

def load_options_list(filepath):
    logger.info("Load a list of options. {}".format(filepath))

    with open(filepath,"r",encoding="UTF-8") as r:
        lines=r.read().splitlines()

    options=[]
    ops=None
    for line in lines:
        if ops is None:
            ops=Options()

        if line=="":
            options.append(ops)
            ops=None
        else:
            ops.append(line)

    return options

def create_dataset(input_dir,num_examples=-1,num_options=4):
    input_ids=torch.load(os.path.join(input_dir,"input_ids.pt"))
    attention_mask=torch.load(os.path.join(input_dir,"attention_mask.pt"))
    token_type_ids=torch.load(os.path.join(input_dir,"token_type_ids.pt"))
    labels=torch.load(os.path.join(input_dir,"labels.pt"))

    indices=torch.empty(input_ids.size(0),dtype=torch.long)
    for i in range(input_ids.size(0)):
        indices[i]=i

    input_ids=input_ids[:,:num_options,:]
    attention_mask=attention_mask[:,:num_options,:]
    token_type_ids=token_type_ids[:,:num_options,:]

    if num_examples>0:
        input_ids=input_ids[:num_examples,:,:]
        attention_mask=attention_mask[:num_examples,:,:]
        token_type_ids=token_type_ids[:num_examples,:,:]
        labels=labels[:num_examples]
        indices=indices[:num_examples]

    return TensorDataset(indices,input_ids,attention_mask,token_type_ids,labels)

def create_text_embeddings(bert_model,options_ids):
    bert_model.eval()

    num_options=options_ids.size(0)

    ret=torch.empty(num_options,512,768).to(device)

    for i in range(num_options):
        input_ids=options_ids[i].unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs=bert_model(input_ids)
            embeddings=bert_model.get_input_embeddings()

            ret[i]=embeddings(input_ids)

    return ret

def create_option_embedding(text_embedding,im_embedding):
    im_embedding_length=im_embedding.size(0)
    text_embedding=text_embedding[:512-im_embedding_length]
    text_embedding[-1]=3    #[SEP]
    option_embedding=torch.cat([text_embedding,im_embedding],dim=0)

    token_type_ids=torch.zeros(512,dtype=torch.long).to(device)
    for i in range(512-im_embedding_length,512):
        token_type_ids[i]=1

    return option_embedding,token_type_ids

def create_inputs_embeds_and_token_type_ids(bert_model,input_ids,indices,options,im_embeddings_dir):
    batch_size=input_ids.size(0)
    num_options=input_ids.size(1)

    inputs_embeds=torch.empty(batch_size,num_options,512,768).to(device)
    inputs_token_type_ids=torch.empty(batch_size,num_options,512,dtype=torch.long).to(device)

    for i in range(batch_size):
        text_embeddings=create_text_embeddings(bert_model,input_ids[i])

        ops=options[indices[i]]
        for j in range(num_options):
            article_name=ops.get(j)
            article_hash=hashing.get_md5_hash(article_name)

            option_embedding=None
            inputs_token_type_ids_tmp=None
            im_features_filepath=os.path.join(im_embeddings_dir,article_hash+".pt")

            if os.path.exists(im_features_filepath):
                im_embedding=torch.load(im_features_filepath,map_location=device).to(device)
                option_embedding,inputs_token_type_ids_tmp=create_option_embedding(text_embeddings[j],im_embedding)
            else:
                option_embedding=text_embeddings[j]
                inputs_token_type_ids_tmp=torch.zeros(512,dtype=torch.long).to(device)

            inputs_embeds[i,j]=option_embedding
            inputs_token_type_ids[i,j]=inputs_token_type_ids_tmp

    return inputs_embeds,inputs_token_type_ids

def train(bert_model,bfmc_model,options,im_embeddings_dir,optimizer,scheduler,dataloader):
    bert_model.eval()
    bfmc_model.train()

    for batch_idx,batch in enumerate(dataloader):
        batch = tuple(t for t in batch)

        inputs = {
            "indices":batch[0].to(device),
            "input_ids": batch[1].to(device),
            "attention_mask": batch[2].to(device),
            "token_type_ids": batch[3].to(device),
            "labels": batch[4].to(device)
        }

        inputs_embeds,inputs_token_type_ids=create_inputs_embeds_and_token_type_ids(
            bert_model,inputs["input_ids"],inputs["indices"],options,im_embeddings_dir
        )

        bfmc_inputs={
            "inputs_embeds":inputs_embeds,
            "attention_mask":inputs["attention_mask"],
            "token_type_ids":inputs_token_type_ids,
            "labels":inputs["labels"]
        }

        # Initialize gradiants
        bfmc_model.zero_grad()
        # Forward propagation
        outputs = bfmc_model(**bfmc_inputs)
        loss = outputs[0]
        # Backward propagation
        loss.backward()
        nn.utils.clip_grad_norm_(bfmc_model.parameters(), 1.0)
        # Update parameters
        optimizer.step()
        scheduler.step()

        if batch_idx%100==0:
            logger.info("Step: {}\tLoss: {}\tlr: {}".format(
                batch_idx,loss.item(),optimizer.param_groups[0]["lr"]))

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def evaluate(bert_model,bfmc_model,options,im_embeddings_dir,dataloader):
    bert_model.eval()
    bfmc_model.eval()

    eval_loss = 0.0
    nb_eval_steps = 0

    preds = None
    correct_labels = None

    for batch_idx,batch in enumerate(dataloader):
        with torch.no_grad():
            batch_size=len(batch)
            batch = tuple(t for t in batch)

            inputs = {
                "indices":batch[0].to(device),
                "input_ids": batch[1].to(device),
                "attention_mask": batch[2].to(device),
                "token_type_ids": batch[3].to(device),
                "labels": batch[4].to(device)
            }

            inputs_embeds,inputs_token_type_ids=create_inputs_embeds_and_token_type_ids(
                bert_model,inputs["input_ids"],inputs["indices"],options,im_embeddings_dir
            )

            bfmc_inputs={
                "inputs_embeds":inputs_embeds,
                "attention_mask":inputs["attention_mask"],
                "token_type_ids":inputs_token_type_ids,
                "labels":inputs["labels"]
            }

            outputs = bfmc_model(**bfmc_inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            correct_labels = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            correct_labels = np.append(
                correct_labels, inputs["labels"].detach().cpu().numpy(), axis=0
            )

    eval_loss = eval_loss / nb_eval_steps
    pred_labels = np.argmax(preds, axis=1)

    accuracy = simple_accuracy(pred_labels, correct_labels)

    return pred_labels,correct_labels,accuracy

def main(batch_size,num_epochs,lr,train_input_dir,dev1_input_dir,im_embeddings_dir,result_save_dir):
    logger.info("Seed: {}".format(SEED))
    logger.info("batch_size: {} num_epochs: {} lr: {}".format(batch_size,num_epochs,lr))

    #Load lists of options.
    logger.info("Load lists of options.")
    train_options=load_options_list(os.path.join(train_input_dir,"options_list.txt"))
    dev1_options=load_options_list(os.path.join(dev1_input_dir,"options_list.txt"))

    #Create dataloaders.
    logger.info("Create train dataset from {}.".format(train_input_dir))
    train_dataset=create_dataset(train_input_dir,num_examples=-1,num_options=4)

    logger.info("Create dev1 dataloader from {}.".format(dev1_input_dir))
    dev1_dataset=create_dataset(dev1_input_dir,num_examples=-1,num_options=20)
    dev1_dataloader=DataLoader(dev1_dataset,batch_size=4,shuffle=False,drop_last=False)

    #Load a pre-trained BERT model.
    logger.info("Load a pre-trained BERT model.")
    bert_model=BertModel.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    bert_model.to(device)

    #Create a BertForMultipleChoice model.
    logger.info("Create a BertForMultipleChoice model.")
    bfmc_model=BertForMultipleChoice.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    bfmc_model.to(device)

    #Create an optimizer and a scheduler.
    num_iterations=len(train_dataset)//batch_size
    total_steps = num_iterations*num_epochs

    optimizer=AdamW(bfmc_model.parameters(),lr=lr,eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    #Create a directory to save the results in.
    os.makedirs(result_save_dir,exist_ok=True)

    logger.info("Start model training.")
    for epoch in range(num_epochs):
        logger.info("===== Epoch {}/{} =====".format(epoch+1,num_epochs))

        train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=False)
        train(bert_model,bfmc_model,train_options,im_embeddings_dir,optimizer,scheduler,train_dataloader)
        pred_labels,correct_labels,accuracy=evaluate(bert_model,bfmc_model,dev1_options,im_embeddings_dir,dev1_dataloader)

        logger.info("Accuracy: {}".format(accuracy))

        #Save model parameters.
        checkpoint_filepath=os.path.join(result_save_dir,"checkpoint_{}.pt".format(epoch+1))
        torch.save(bfmc_model.state_dict(),checkpoint_filepath)

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
    parser.add_argument("--num_epochs",type=int,default=5)
    parser.add_argument("--lr",type=float,default=2.5e-5)
    parser.add_argument("--train_input_dir",type=str,default="~/EncodedTextTohoku/Train")
    parser.add_argument("--dev1_input_dir",type=str,default="~/EncodedTextTohoku/Dev1")
    parser.add_argument("--im_embeddings_dir",type=str,default="~/FasterRCNNEmbeddings/Projected")
    parser.add_argument("--result_save_dir",type=str,default="./OutputDir")

    args=parser.parse_args()

    main(
        args.batch_size,
        args.num_epochs,
        args.lr,
        args.train_input_dir,
        args.dev1_input_dir,
        args.im_embeddings_dir,
        args.result_save_dir
    )
