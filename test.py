import os
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pandas as pd

import tensorflow as tf


from caption.caption import CaptionGenerator
from config.config import Config
from models.models_new import ShowAttendTell
from utils.dataprovider import DataProvider
from utils.vocabulary import Vocabulary

from utils.metrics.pycocoevalcap.eval import COCOEvalCap

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')


def main(argv):
    print("Testing the model ...")
    config = Config()
    config.beam_size = FLAGS.beam_size
    config.phase = 'test'

    if not os.path.exists(config.test_result_dir):
        os.mkdir(config.test_result_dir)
    print("Building the vocabulary...")
    vocabulary = Vocabulary(config.vocabulary_size)
    vocabulary.load(config.vocabulary_file)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))

    
    test_data = DataProvider(config)
    test_gt_coco = test_data.returncoco()
    model = ShowAttendTell(config)
    model.build()

    with tf.Session() as sess:
        model.setup_graph_from_checkpoint(sess, config.caption_checkpoint_dir)
        tf.get_default_graph().finalize()

        captiongen = CaptionGenerator(model,
                                   vocabulary,
                                   config.beam_size,
                                   config.max_caption_length,
                                   config.batch_size)
        
        # Generate the captions for the images
        results = []
        idx = 0
        for k in tqdm(list(range(test_data.num_batches)), desc='batch'):
            batch,images = test_data.next_batch_and_images()
            caption_data = captiongen.beam_search(sess, images,vocabulary)

            fake_cnt = 0 if k<test_data.num_batches-1 \
                         else test_data.fake_count
            for l in range(test_data.batch_size-fake_cnt):
                word_idxs = caption_data[l][0].sentence
                score = caption_data[l][0].score
                caption = vocabulary.get_sentence(word_idxs)
                results.append({'image_id': test_data.image_ids[idx],
                                'caption': caption})
                idx += 1

                # Save the result in an image file, if requested
                if config.save_test_result_as_image:
                    image_file = batch[l]
                    image_name = image_file.split(os.sep)[-1]
                    image_name = os.path.splitext(image_name)[0]
                    img = plt.imread(image_file)
                    plt.switch_backend('agg')
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(caption)
                    plt.savefig(os.path.join(config.test_result_dir,
                                             image_name+'_result.png'))

        fp = open(config.test_result_file, 'wb')
        json.dump(results, fp)
        fp.close()

        # Evaluate these captions
        test_result_coco = test_gt_coco.loadRes(config.test_result_file)
        scorer = COCOEvalCap(test_gt_coco, test_result_coco)
        scorer.evaluate()
    print("Evaluation complete.")



if __name__ == '__main__':
    tf.app.run()
