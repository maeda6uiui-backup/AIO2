import argparse
import cv2
import glob
import logging
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm

from detectron2.utils.logger import setup_logger
setup_logger()

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import hashing

logging_fmt = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_region_features_single(raw_image,predictor):
    with torch.no_grad():
        raw_height,raw_width=raw_image.shape[:2]

        image=predictor.aug.get_transform(raw_image).apply_image(raw_image)
        image=torch.as_tensor(image.astype("float32").transpose(2,0,1))
        inputs=[{"image":image,"height":raw_height,"width":raw_width}]
        images=predictor.model.preprocess_image(inputs)

        model=predictor.model

        #Extract features from the backbone
        features=model.backbone(images.tensor)
        #Generate proposals
        proposals,_=model.proposal_generator(images,features)
        #RoI align
        box_features=model.roi_heads.box_pooler(
            [features[f] for f in features if f!="p6"],
            [p.proposal_boxes for p in proposals]
        )
        #Get features from fc2
        box_features=model.roi_heads.box_head(box_features)

        return box_features #(Number of RoIs,1024)

def get_region_features(raw_images,predictor):
    lst=list()
    for raw_image in raw_images:
        features=get_region_features_single(raw_image,predictor)
        lst.append(features)

    #Concatenate tensors.
    if len(lst)==0:
        return torch.zeros(0,0).to(device)

    dimension=lst[0].size(1)
    features=torch.empty(0,dimension).to(device)
    for f in lst:
        features=torch.cat([features,f],dim=0)
    
    return features

def create_image_embeddings(image_root_dir,article_list_filepath,embeddings_save_dir,predictor):
    os.makedirs(embeddings_save_dir,exist_ok=True)

    df=pd.read_csv(article_list_filepath,encoding="utf_8",sep="\t")
    for row in tqdm(df.values):
        article_name,sec1,sec2=row[:3]
        image_dir=os.path.join(image_root_dir,str(sec1),str(sec2))

        #Load images.
        pathname=os.path.join(image_dir,"*")
        image_files=glob.glob(pathname)
        images=list()
        for image_file in image_files:
            image=cv2.imread(image_file)
            images.append(image)

        features=get_region_features(images,predictor)

        article_hash=hashing.get_md5_hash(article_name)
        save_filepath=os.path.join(embeddings_save_dir,article_hash+".pt")

        torch.save(features,save_filepath)

def main(image_root_dir,article_list_filepath,model_name,embeddings_save_dir):
    logger.info("Creating a DefaultPredictor.")
    cfg=get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.7
    cfg.MODEL.DEVICE="cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url(model_name)
    predictor=DefaultPredictor(cfg)
    
    logger.info("Start creating image embeddings.")
    create_image_embeddings(image_root_dir,article_list_filepath,embeddings_save_dir,predictor)
    logger.info("Finished creating image embeddings.")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--image_root_dir",type=str,default="~/WikipediaImages/Images")
    parser.add_argument("--article_list_filepath",type=str,default="./article_list.txt")
    parser.add_argument("--model_name",type=str,default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument("--embeddings_save_dir",type=str,default="~/FasterRCNNEmbeddings")
    args=parser.parse_args()

    main(
        args.image_root_dir,
        args.article_list_filepath,
        args.model_name,
        args.embeddings_save_dir
    )
