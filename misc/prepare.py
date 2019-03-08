import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.metrics.cocoset import COCO
from utils.vocabulary import Vocabulary

from config.config import Config

def prepare_train_data(config):
    """ Prepare the data for training the model. """
    if not os.path.exists(config.prepare_annotation_dir):
        os.mkdir(config.prepare_annotation_dir)
    coco = COCO(config, config.train_caption_file, config.val_caption_file)
    
    print("Building the vocabulary...")
    vocabulary = Vocabulary(config.vocabulary_size)
    if not os.path.exists(config.vocabulary_file):
        coco.filter_by_cap_len(config.max_caption_length)
        vocabulary.build(coco.all_captions())
        vocabulary.save(config.vocabulary_file)
        vocabulary.save_counts(config.word_count_file)
    else:
        vocabulary.load(config.vocabulary_file)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))

    
    print("Processing the captions...")
    if not os.path.exists(config.train_csv_file):
                    
        coco.filter_by_words(set(vocabulary.words))
        captions = [coco.anns[ann_id]['caption'] for ann_id in coco.anns]
        image_ids = [coco.anns[ann_id]['image_id'] for ann_id in coco.anns]
        image_files = [ 
            os.path.join(config.dataset_image_dir,
            'train' if coco.imgs[image_id]['file_name'].find('train2014')>=0 else 'val',
            coco.imgs[image_id]['file_name'])
                        for image_id in image_ids ] 
        annotations = pd.DataFrame({'image_id': image_ids,
                                    'image_file': image_files,
                                    'caption': captions})
        annotations.to_csv(config.train_csv_file)
    else:
        annotations = pd.read_csv(config.train_csv_file)
        captions = annotations['caption'].values
        image_ids = annotations['image_id'].values
        image_files = annotations['image_file'].values


if __name__=='__main__':
    config = Config()
    prepare_train_data(config)