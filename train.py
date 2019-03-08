import os
from tqdm import tqdm

import tensorflow as tf

from config.config import Config
from models.models_new import ShowAttendTell
from utils.trainprovider import TrainDataProvider


FLAGS = tf.app.flags.FLAGS


tf.flags.DEFINE_boolean('load_checkpoint', True,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint')
tf.flags.DEFINE_integer('number_of_steps', None,
                        'Train numer of steps')

tf.flags.DEFINE_boolean('load_cnn', True,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './models/vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

def main(argv):
    config = Config()
    config.train_cnn = FLAGS.train_cnn
    config.phase = 'train'
    if not FLAGS.number_of_steps is None:
        config.number_of_steps = FLAGS.number_of_steps

    is_load_checkpoint = FLAGS.load_checkpoint
    checkpoint_dir = config.checkpoint_dir
    if not os.path.exists(config.summary_dir):
        os.makedirs(config.summary_dir)  
    if not os.path.exists(config.results_dir):
        os.makedirs(config.results_dir)
    if not os.path.exists(checkpoint_dir):
        is_load_checkpoint = False
        os.makedirs(checkpoint_dir)  

    dataprovider = TrainDataProvider(config)
    model = ShowAttendTell(config)
    model.build(dataprovider)  
    
    start=0
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(config.summary_dir,sess.graph)
        saver = tf.train.Saver(max_to_keep=None)
        if is_load_checkpoint:
            start=model.setup_graph_from_checkpoint(sess, config.caption_checkpoint_dir)
            tf.get_default_graph().finalize()
        else:
            sess.run(tf.global_variables_initializer())
            if FLAGS.load_cnn:
                model.load_cnn(sess, FLAGS.cnn_model_file)
            tf.get_default_graph().finalize()

        print("Training the model...")

        coord = tf.train.Coordinator()
        threadlist = tf.train.start_queue_runners(coord=coord)
        print(start)
        start = start // 32
        count = tqdm(range(start, config.number_of_steps), total=len(list(range(start, config.number_of_steps))), ncols=60)
        for _ in count:
            total_loss, summary, global_step = sess.run([model.opt_op,
                                                    model.summary,
                                                    model.global_step])
            train_writer.add_summary(summary, global_step)
            tf.logging.info('global step %d: loss = %.4f', global_step, total_loss)
            
            if global_step % 10 == 0:
                count.set_description('global step %d: loss = %.4f' % (global_step, total_loss))
            if (global_step + 1) % config.save_checkpoint_period == 0:
                saver.save(sess,config.caption_checkpoint_path,global_step) 


        saver.save(sess,config.caption_checkpoint_path,global_step)  
        coord.request_stop()
        coord.join(threadlist)
        train_writer.close()
           
        
    print("Training complete.")

if __name__ == '__main__':
    tf.app.run()
