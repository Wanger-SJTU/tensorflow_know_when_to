import os
import math
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from metrics.cocoset import COCO


class ImageLoader(object):
    def __init__(self, config, mean_file=None):
        self.bgr = True
        self.scale_shape = np.array(
                [config.image_height, config.image_width], np.int32)
        self.crop_shape = np.array(
                [config.image_height, config.image_width], np.int32)
        if mean_file is None:
            self.mean = np.array([122.679,116.669,104.007],
                dtype=np.float32)  # RGB
        else:
            self.mean = np.load(mean_file).mean(1).mean(1)

    def load_image(self, image_file):
        """ Load and preprocess an image. """
        image = cv2.imread(image_file)

        if self.bgr:
            temp = image.swapaxes(0, 2)
            temp = temp[::-1]
            image = temp.swapaxes(0, 2)

        image = cv2.resize(image, (self.scale_shape[0], self.scale_shape[1]))
        offset = (self.scale_shape - self.crop_shape) / 2
        offset = offset.astype(np.int32)
        image = image[offset[0]:offset[0]+self.crop_shape[0],
                      offset[1]:offset[1]+self.crop_shape[1]]
        image = image - self.mean
        return image

    def load_images(self, image_files):
        """ Load and preprocess a list of images. """
        images = []
        for image_file in image_files:
            images.append(self.load_image(image_file))
        images = np.array(images, np.float32)
        return images


class DataProvider(object):
    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size
        self.is_eval = True if config.phase == 'eval' else False
        self.is_test = True if config.phase == 'test' else False
        self.is_infer = True if config.phase == 'infer' else False

        if self.is_eval:
            self.coco = COCO(config=config,
                            first_ann_file=config.valpart_eval_json_file)
            image_ids = list(self.coco.imgs.keys())
            image_files = [os.path.join(config.eval_image_dir,
                                self.coco.imgs[image_id]['file_name'])
                                for image_id in image_ids]
        elif self.is_test:
            self.coco = COCO(config=config,
                            first_ann_file=config.valpart_test_json_file)
            image_ids = list(self.coco.imgs.keys())
            image_files = [os.path.join(config.test_image_dir,
                                self.coco.imgs[image_id]['file_name'])
                                for image_id in image_ids]
        elif self.is_infer:
            self.coco = None
            files = os.listdir(config.infer_image_dir)
            image_files = [os.path.join(config.infer_image_dir, f) for f in files
                    if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]
            image_ids = list(range(len(image_files)))
        else:
            raise RuntimeError("Error from dataprovider!!!")
        self.setup(image_ids,image_files)
        
    def returncoco(self):
        return self.coco

    def setup(self, image_ids, image_files):
        self.image_ids = np.array(image_ids)
        self.image_files = np.array(image_files)

        """ Setup the dataset. """
        self.count = len(self.image_ids)
        self.num_batches = int(np.ceil(self.count * 1.0 / self.batch_size))
        self.fake_count = self.num_batches * self.batch_size - self.count
        self.idxs = list(range(self.count))
        self.reset()

    def reset(self):
        """ Reset the dataset. """
        self.current_idx = 0


    def next_batch_and_images(self):
        image_files = self.next_batch()

        self.image_loader = ImageLoader(self.config)
        images = self.image_loader.load_images(image_files)

        return image_files, images

    def next_batch(self):
        """ Fetch the next batch. """
        assert self.has_next_batch()

        if self.has_full_next_batch():
            start, end = self.current_idx, \
                         self.current_idx + self.batch_size
            current_idxs = self.idxs[start:end]
        else:
            start, end = self.current_idx, self.count
            current_idxs = self.idxs[start:end] + \
                           list(np.random.choice(self.count, self.fake_count))

        image_files = self.image_files[current_idxs]
        self.current_idx += self.batch_size
        return image_files

    def has_next_batch(self):
        """ Determine whether there is a batch left. """
        return self.current_idx < self.count

    def has_full_next_batch(self):
        """ Determine whether there is a full batch left. """
        return self.current_idx + self.batch_size <= self.count
