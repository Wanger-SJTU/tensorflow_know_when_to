import os
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

tf.flags.DEFINE_integer('beam_size', 5,
                        'The size of beam search for caption generation')


def main(argv):
    print("Evaluating the model ...")
    config = Config()
    config.beam_size = FLAGS.beam_size
    config.phase = 'eval'
    if not os.path.exists(config.eval_result_dir):
        os.mkdir(config.eval_result_dir)
    print("Building the vocabulary...")
    vocabulary = Vocabulary(config.vocabulary_size)
    vocabulary.load(config.vocabulary_file)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))

    
    eval_data = DataProvider(config)
    eval_gt_coco = eval_data.returncoco()
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
        for k in tqdm(list(range(eval_data.num_batches)), desc='batch'):
            batch,images = eval_data.next_batch_and_images()
            caption_data = captiongen.beam_search(sess, images,vocabulary)

            fake_cnt = 0 if k<eval_data.num_batches-1 \
                         else eval_data.fake_count
            for l in range(eval_data.batch_size-fake_cnt):
                word_idxs = caption_data[l][0].sentence
                score = caption_data[l][0].score
                caption = vocabulary.get_sentence(word_idxs)
                results.append({'image_id': eval_data.image_ids[idx],
                                'caption': caption})
                idx += 1

                # Save the result in an image file, if requested
                if config.save_eval_result_as_image:
                    image_file = batch[l]
                    image_name = image_file.split(os.sep)[-1]
                    image_name = os.path.splitext(image_name)[0]
                    img = plt.imread(image_file)
                    plt.switch_backend('agg')
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(caption)
                    plt.savefig(os.path.join(config.eval_result_dir,
                                             image_name+'_result.jpg'))

        fp = open(config.eval_result_file, 'wb')
        json.dump(results, fp)
        fp.close()

        # Evaluate these captions
        eval_result_coco = eval_gt_coco.loadRes(config.eval_result_file)
        scorer = COCOEvalCap(eval_gt_coco, eval_result_coco)
        scorer.evaluate()
    print("Evaluation complete.")



if __name__ == '__main__':
    tf.app.run()
