"""Helper functions for Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import tensorflow as tf

def distort_image(image, thread_id):
    """Perform random distortions on an image.

    Args:
        image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).
        thread_id: Preprocessing thread id used to select the ordering of color
            distortions. There should be a multiple of 2 preprocessing threads.

    Returns:
        distorted_image: A float32 Tensor of shape [height, width, 3] with values in
            [0, 1].
    """
    # Randomly flip horizontally.
    with tf.name_scope("flip_horizontal", values=[image]):
        image = tf.image.random_flip_left_right(image)

    # Randomly distort the colors based on thread id.
    color_ordering = thread_id % 2
    with tf.name_scope("distort_color", values=[image]):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.032)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.032)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)

    return image


def process_image_(encoded_image,
                height,
                width,
                resize_height,
                resize_width,
                thread_id=0,
                image_format="jpeg"):
    """Decode an image, resize and apply random distortions.

    In training, images are distorted slightly differently depending on thread_id.

    Args:
        encoded_image: String Tensor containing the image.
        is_training: Boolean; whether preprocessing for training or eval.
        height: Height of the output image.
        width: Width of the output image.
        resize_height: If > 0, resize height before crop to final dimensions.
        resize_width: If > 0, resize width before crop to final dimensions.
        thread_id: Preprocessing thread id used to select the ordering of color
            distortions. There should be a multiple of 2 preprocessing threads.
        image_format: "jpeg" or "png".

    Returns:
        A float32 Tensor of shape [height, width, 3] with values in [-1, 1].

    Raises:
        ValueError: If image_format is invalid.
    """
    # Helper function to log an image summary to the visualizer. Summaries are
    # only logged in thread 0.

    # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
    with tf.name_scope("decode", values=[encoded_image]):
        if image_format == "jpeg":
            image = tf.image.decode_jpeg(encoded_image, channels=3)
        elif image_format == "png":
            image = tf.image.decode_png(encoded_image, channels=3)
        else:
            raise ValueError("Invalid image format: %s" % image_format)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Resize image.
    assert (resize_height > 0) == (resize_width > 0)
    if resize_height:
        image = tf.image.resize_images(image,
                            size=[resize_height, resize_width],
                            method=tf.image.ResizeMethod.BILINEAR)

    # Crop to final dimensions.
    image = tf.random_crop(image, [height, width, 3])



    # Randomly distort the image.
    #image = distort_image(image, thread_id)


    # Rescale to [-1,1] instead of [0, 1]
    image = tf.multiply(image, 255.0)# to 256 scale
    image = tf.subtract(image, [122.679,116.669,104.007])

    return image


class TrainDataProvider(object):
    def __init__(self,config):
        # Reader for the input data.
        self.reader = tf.TFRecordReader()
        self.config = config


    def generate_batch_data(self):

        input_queue = self.prefetch_input_data(
                self.reader,
                self.config.input_file_pattern,
                batch_size=self.config.batch_size,
                values_per_shard=self.config.values_per_input_shard,
                input_queue_capacity_factor=self.config.input_queue_capacity_factor,
                num_reader_threads=self.config.num_input_reader_threads)

        # Image processing and random distortion. Split across multiple threads
        # with each thread applying a slightly different distortion.
        assert self.config.num_preprocess_threads % 2 == 0
        
        images_captions_masks = []
        for thread_id in range(self.config.num_preprocess_threads):
            serialized_sequence_example = input_queue.dequeue()
            encoded_image, caption, mask, cls_lbl = self.parse_sequence_example(
                        serialized_sequence_example,
                        image_feature=self.config.image_feature_name,
                        caption_feature=self.config.caption_feature_name,
                        mask_feature = self.config.mask_feature_name,
                        cls_lbl_feature=self.config.caption_lbl_name)
            image = process_image_(encoded_image,
                            height=self.config.image_height,
                            width=self.config.image_width,
                            resize_height=self.config.image_height,
                            resize_width=self.config.image_width,
                            image_format=self.config.image_format)

            images_captions_masks.append([image, caption, mask, cls_lbl])

        # Batch inputs.
        queue_capacity = (3 * self.config.num_preprocess_threads *
                                                self.config.batch_size)
        image_batch, caption_batch, mask_batch, cls_lbl_batch  = (
                    self.batch_with_dynamic_pad(images_captions_masks,
                                        batch_size=self.config.batch_size,
                                        queue_capacity=queue_capacity))

        return image_batch, caption_batch, mask_batch, cls_lbl_batch


    def parse_sequence_example(self,serialized, image_feature, 
                                caption_feature, mask_feature,cls_lbl_feature):
        """Parses a tensorflow.SequenceExample into an image and caption.

        Args:
            serialized: A scalar string Tensor; a single serialized SequenceExample.
            image_feature: Name of SequenceExample context feature containing image
                data.
            caption_feature: Name of SequenceExample feature list containing integer
                captions.

        Returns:
            encoded_image: A scalar string Tensor containing a JPEG encoded image.
            caption: A 1-D uint64 Tensor with dynamically specified length.
        """
        context, sequence = tf.parse_single_sequence_example(
                serialized,
                context_features={
                        image_feature: tf.FixedLenFeature([], dtype=tf.string)
                },
                sequence_features={
                        caption_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64),
                        mask_feature: tf.FixedLenSequenceFeature([],dtype=tf.float32),
                        cls_lbl_feature:tf.FixedLenSequenceFeature([], dtype=tf.int64)
                })

        encoded_image = context[image_feature]
        caption = sequence[caption_feature]
        mask = sequence[mask_feature]
        cls_lbl = sequence[cls_lbl_feature]
        return encoded_image, caption, mask,cls_lbl


    def prefetch_input_data(self, reader,
                            file_pattern,
                            batch_size,
                            values_per_shard,
                            input_queue_capacity_factor=16,
                            num_reader_threads=1,
                            shard_queue_name="filename_queue",
                            value_queue_name="input_queue"):
        """Prefetches string values from disk into an input queue.

        In training the capacity of the queue is important because a larger queue
        means better mixing of training examples between shards. The minimum number of
        values kept in the queue is values_per_shard * input_queue_capacity_factor,
        where input_queue_memory factor should be chosen to trade-off better mixing
        with memory usage.
    
        Args:
            reader: Instance of tf.ReaderBase.
            file_pattern: Comma-separated list of file patterns (e.g.
                    /tmp/train_data-?????-of-00100).
            batch_size: Model batch size used to determine queue capacity.
            values_per_shard: Approximate number of values per shard.
            input_queue_capacity_factor: Minimum number of values to keep in the queue
                in multiples of values_per_shard. See comments above.
            num_reader_threads: Number of reader threads to fill the queue.
            shard_queue_name: Name for the shards filename queue.
            value_queue_name: Name for the values input queue.

        Returns:
            A Queue containing prefetched string values.
        """
        data_files = []
        for pattern in file_pattern.split(","):
            data_files.extend(tf.gfile.Glob(pattern))
        if not data_files:
            tf.logging.fatal("Found no input files matching %s", file_pattern)
        else:
            tf.logging.info("Prefetching values from %d files matching %s",
                                        len(data_files), file_pattern)


        filename_queue = tf.train.string_input_producer(
                    data_files, shuffle=True, capacity=16, name=shard_queue_name)
        min_queue_examples = values_per_shard * input_queue_capacity_factor
        capacity = min_queue_examples + 100 * batch_size
        values_queue = tf.RandomShuffleQueue(
                capacity=capacity,
                min_after_dequeue=min_queue_examples,
                dtypes=[tf.string],
                name="random_" + value_queue_name)


        enqueue_ops = []
        for _ in range(num_reader_threads):
            _, value = reader.read(filename_queue)
            enqueue_ops.append(values_queue.enqueue([value]))
        tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
                values_queue, enqueue_ops))

        return values_queue


    def batch_with_dynamic_pad(self,
                                images_captions_masks,
                                batch_size,
                                queue_capacity,
                                add_summaries=False):
        """Batches input images and captions.

        This function splits the caption into an input sequence and a 
        target sequence, where the target sequence is the input sequence 
        right-shifted by 1. Input and target sequences are batched and 
        padded up to the maximum length of sequences in the batch. 
        A mask is created to distinguish real words from padding words.

        Example:
            Actual captions in the batch ('-' denotes padded character):
            [
                [ 1 2 5 4 5 ],
                [ 1 2 3 4 - ],
                [ 1 2 3 - - ],
            ]

            input_seqs:
            [
                [ 1 2 3 4 ],
                [ 1 2 3 - ],
                [ 1 2 - - ],
            ]

            target_seqs:
            [
                [ 2 3 4 5 ],
                [ 2 3 4 - ],
                [ 2 3 - - ],
            ]

            mask:
            [
                [ 1 1 1 1 ],
                [ 1 1 1 0 ],
                [ 1 1 0 0 ],
            ]

        Args:
            images_and_captions: A list of pairs [image, caption], where image 
            is a Tensor of shape [height, width, channels] and caption is a 
            1-D Tensor of any length. Each pair will be processed and added to 
            the queue in a separate thread.
            
            batch_size: Batch size.
            queue_capacity: Queue capacity.
            add_summaries: If true, add caption length summaries.

        Returns:
            images: A Tensor of shape [batch_size, height, width, channels].
            input_seqs: An int32 Tensor of shape [batch_size, padded_length].
            target_seqs: An int32 Tensor of shape [batch_size, padded_length].
            mask: An int32 0/1 Tensor of shape [batch_size, padded_length].
        """
        enqueue_list = []
        for image, caption, mask, cls_lbl in images_captions_masks:
             enqueue_list.append([image, caption, mask, cls_lbl])

        image_batch, caption_batch, mask_batch, cls_lbl_batch = tf.train.batch_join(
                enqueue_list,
                batch_size=batch_size,
                capacity=queue_capacity,
                dynamic_pad=True,
                name="batch_and_pad")

        if add_summaries:
            lengths = tf.add(tf.reduce_sum(mask_batch, 1), 1)
            tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths))
            tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths))
            tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths))

        return image_batch, caption_batch, mask_batch,cls_lbl_batch
