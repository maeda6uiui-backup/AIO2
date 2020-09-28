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
from transformers import BertForMultipleChoice,AdamW,get_linear_schedule_with_warmup

#Fix the seed.
SEED=1234
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

def create_dataset(input_dir,num_examples=-1,num_options=4):
    input_ids=torch.load(os.path.join(input_dir,"input_ids.pt"))
    attention_mask=torch.load(os.path.join(input_dir,"attention_mask.pt"))
    token_type_ids=torch.load(os.path.join(input_dir,"token_type_ids.pt"))
    labels=torch.load(os.path.join(input_dir,"labels.pt"))

    input_ids=input_ids[:,:num_options,:]
    attention_mask=attention_mask[:,:num_options,:]
    token_type_ids=token_type_ids[:,:num_options,:]

    if num_examples>0:
        input_ids=input_ids[:num_examples,:,:]
        attention_mask=attention_mask[:num_examples,:,:]
        token_type_ids=token_type_ids[:num_examples,:,:]
        labels=labels[:num_examples]

    return TensorDataset(input_ids,attention_mask,token_type_ids,labels)

def simple_accuracy(pred_labels, correct_labels):
    return (pred_labels == correct_labels).mean()

def evaluate(classifier_model,dataloader):
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

            if preds is None:
                preds = logits.detach().cpu().numpy()
                correct_labels = bert_inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                correct_labels = np.append(
                    correct_labels, bert_inputs["labels"].detach().cpu().numpy(), axis=0
                )

    pred_labels = np.argmax(preds, axis=1)
    accuracy = simple_accuracy(pred_labels, correct_labels)

    return pred_labels,correct_labels,accuracy

def main(test_input_dir,model_dir,test_upper_bound,result_save_dir):
    logger.info("Seed: {}".format(SEED))

    #Create a dataloader.
    logger.info("Create test dataloader from {}.".format(test_input_dir))
    test_dataset=create_dataset(test_input_dir,num_examples=-1,num_options=20)
    test_dataloader=DataLoader(test_dataset,batch_size=4,shuffle=False,drop_last=True)

    #Create a classifier model.
    logger.info("Create a classifier model.")
    classifier_model=BertForMultipleChoice.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    classifier_model.to(device)

    #Create a directory to save the results in.
    logger.info("Results will be saved in {}.".format(result_save_dir))
    os.makedirs(result_save_dir,exist_ok=True)

    logger.info("Start model evaluation.")
    for i in range(test_upper_bound):
        model_filepath=os.path.join(model_dir,"checkpoint_{}.pt".format(i+1))
        logger.info("Load model parameters from {}.".format(model_filepath))

        parameters=torch.load(model_filepath,map_location=device)
        classifier_model.load_state_dict(parameters)

        pred_labels,correct_labels,accuracy=evaluate(classifier_model,test_dataloader)
        logger.info("Accuracy: {}".format(accuracy))

        #Save results as text files.
        res_filepath=os.path.join(result_save_dir,"result_test_{}.txt".format(i+1))
        labels_filepath=os.path.join(result_save_dir,"labels_test_{}.txt".format(i+1))

        with open(res_filepath,"w") as w:
            w.write("Accuracy: {}\n".format(accuracy))

        with open(labels_filepath,"w") as w:
            for pred_label,correct_label in zip(pred_labels,correct_labels):
                w.write("{} {}\n".format(pred_label,correct_label))

    logger.info("Finished model evaluation.")

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--test_input_dir",type=str,default="~/EncodedTextTohoku/Dev2")
    parser.add_argument("--model_dir",type=str,default="./OutputDir/Baseline")
    parser.add_argument("--test_upper_bound",type=int,default=20)
    parser.add_argument("--result_save_dir",type=str,default="./OutputDir/Baseline")

    args=parser.parse_args()

    main(
        args.test_input_dir,
        args.model_dir,
        args.test_upper_bound,
        args.result_save_dir
    )
