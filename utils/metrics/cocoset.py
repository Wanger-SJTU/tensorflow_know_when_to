
# Rewrite by Zhou Shibin zhoushibin@cumt.edu.cn

import json
import pdb
import time
import os
from tqdm import tqdm
from nltk.tokenize import word_tokenize

class COCO(object):
    def __init__(self, config=None, first_ann_file=None,second_ann_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.config = config

        self.dataset = {}
        self.anns = []
        self.imgToAnns = {}
        self.imgs = {}
        
        if not first_ann_file is None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(first_ann_file, 'r'))
            print('Done (t=%0.2fs)'%(time.time()- tic))
            self.dataset['images']= dataset['images']
            self.dataset['annotations'] = dataset['annotations']
            if 'classifications' in dataset.keys():
                self.dataset['cls_lbls'] = dataset['classifications']
            if not second_ann_file is None:
                dataset_second = json.load(open(second_ann_file, 'r'))
                self.split_second_ann(dataset_second)
            print('Done (t=%0.2fs)'%(time.time()- tic))

            self.process_dataset()
            self.createIndex()

    def split_second_ann(self, dataset):

        train_img = {}
        eval_img = {}
        test_img = {}

        traindataset = {}
        evaldataset = {}
        testdataset = {}

        count = 0

        for img in dataset['images']:
            if count < self.config.eval_img_nums:
                eval_img[img['id']] = eval_img.get(img['id'],0) + 1

            elif count < self.config.eval_img_nums + self.config.test_img_nums:
                test_img[img['id']] = test_img.get(img['id'],0) + 1
            else:
                train_img[img['id']] = train_img.get(img['id'],0) + 1
            count += 1
        

        traindataset['annotations'] = \
            [ann for ann in dataset['annotations'] \
            if ann['image_id'] in train_img.keys()]        
        traindataset['images'] = \
            [ann for ann in dataset['images'] \
            if ann['id'] in train_img.keys()]
        
        evaldataset['annotations'] = \
            [ann for ann in dataset['annotations'] \
            if ann['image_id'] in eval_img.keys()]
        evaldataset['images'] = \
            [ann for ann in dataset['images'] \
            if ann['id'] in eval_img.keys()]
        
        testdataset['annotations'] = \
            [ann for ann in dataset['annotations'] \
            if ann['image_id'] in test_img.keys()]
        testdataset['images'] = \
            [ann for ann in dataset['images'] \
            if ann['id'] in test_img.keys()]
        

        if 'classifications' in dataset.keys():
            traindataset['cls_lbls'] = \
                [ann for ann in dataset['classifications'] \
                if ann['image_id'] in train_img.keys()]
            evaldataset['cls_lbls'] = \
                [ann for ann in dataset['classifications'] \
                if ann['image_id'] in eval_img.keys()]
            testdataset['cls_lbls'] = \
                [ann for ann in dataset['classifications'] \
                if ann['image_id'] in test_img.keys()]
            self.dataset['cls_lbls'].extend(traindataset['cls_lbls'])

        fp = open(self.config.valpart_train_json_file, 'w')
        json.dump(traindataset, fp)
        fp.close()

        fp = open(self.config.valpart_eval_json_file, 'w')
        json.dump(evaldataset, fp)
        fp.close()

        fp = open(self.config.valpart_test_json_file, 'w')
        json.dump(testdataset, fp)
        fp.close()

        self.dataset['images'].extend(traindataset['images'])
        self.dataset['annotations'].extend(traindataset['annotations'])
        

       
    def createIndex(self):
        # create index
        print('creating index...')
        anns = {}
        imgToAnns = {}
        imgToCls_lbls = {}
        imgs = {}
        cls_lbls = {}

        if 'annotations' in self.dataset:
            imgToAnns = {ann['image_id']: [] for ann in self.dataset['annotations']}
            anns =      {ann['id']:       [] for ann in self.dataset['annotations']}
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']] += [ann]
                anns[ann['id']] = ann

        if 'cls_lbls' in self.dataset:
            imgToCls_lbls = {ann['image_id']: [] for ann in self.dataset['cls_lbls']}
            cls_lbls =      {ann['id']:       [] for ann in self.dataset['cls_lbls']}
            for lbl in self.dataset['cls_lbls']:
                imgToCls_lbls[lbl['image_id']] += [lbl]
                cls_lbls[lbl['id']] = lbl

        if 'images' in self.dataset:
            imgs      = {im['id']: {} for im in self.dataset['images']}
            for img in self.dataset['images']:
                imgs[img['id']] = img



        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.imgs = imgs
        if 'cls_lbls' in self.dataset:
            self.imgToCls_lbls = imgToCls_lbls
            self.cls_lbls = cls_lbls


    def getImgIds(self):
        ids = self.imgs.keys()
        return list(ids)

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...     ')
        tic = time.time()
        anns    = json.load(open(resFile))
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
               'Results do not correspond to current coco set'
        assert 'caption' in anns[0]
        imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
        res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
        for id, ann in enumerate(anns):
            ann['id'] = id+1
        print('DONE (t=%0.2fs)'%(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res


    def process_dataset(self):
        for ann in self.dataset['annotations']:
            q = ann['caption'].lower()
            if q[-1]!='.':
                q = q + '.'
            ann['caption'] = q

    def filter_by_cap_len(self, max_cap_len):
        print("Filtering the captions by length...")
        keep_ann = {}
        keep_img = {}
        for ann in tqdm(self.dataset['annotations']):
            if len(word_tokenize(ann['caption'])) <= max_cap_len:
                keep_ann[ann['id']] = keep_ann.get(ann['id'], 0) + 1
                keep_img[ann['image_id']] = keep_img.get(ann['image_id'], 0) + 1

        self.dataset['annotations'] = \
            [ann for ann in self.dataset['annotations'] \
            if keep_ann.get(ann['id'],0)>0]
        self.dataset['images'] = \
            [img for img in self.dataset['images'] \
            if keep_img.get(img['id'],0)>0]

        self.createIndex()

    def filter_by_words(self, vocab):
        print("Filtering the captions by words...")
        keep_ann = {}
        keep_img = {}
        for ann in tqdm(self.dataset['annotations']):
            keep_ann[ann['id']] = 1
            words_in_ann = word_tokenize(ann['caption'])
            for word in words_in_ann:
                if word not in vocab:
                    keep_img[ann['image_id']] = \
                        keep_img.get(ann['image_id'], 0) - 1 # bug fix 2018.9.18
                    keep_ann[ann['id']] = 0
                    break
            keep_img[ann['image_id']] = keep_img.get(ann['image_id'], 0) + 1

        self.dataset['annotations'] = \
            [ann for ann in self.dataset['annotations'] \
            if keep_ann.get(ann['id'],0)>0]
        self.dataset['images'] = \
            [img for img in self.dataset['images'] \
            if keep_img.get(img['id'],0)>0]

        if 'cls_lbls' in self.dataset.keys():
            self.dataset['cls_lbls'] = \
                [lbl for lbl in self.dataset['cls_lbls'] \
                if keep_ann.get(lbl['id'],0)>0]
        self.createIndex()

    def all_captions(self):
        return [ann['caption'] for ann_id, ann in self.anns.items()]
