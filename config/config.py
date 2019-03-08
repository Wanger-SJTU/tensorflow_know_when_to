
class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):

        # Changed more
        self.number_of_steps = 1000000//32 #Number of training steps.
        self.batch_size = 32
        self.save_checkpoint_period = 1000

        # Important text generation parameters
        self.max_caption_length = 20
        self.vocabulary_size = 5000

        self.input_file_pattern='./datasets/shard/train-?????-of-00256'
        # Number of threads for image preprocessing. Should be a multiple of 2.
        self.num_preprocess_threads = 4
        self.num_input_reader_threads = 2

        self.values_per_input_shard = 2300
        self.input_queue_capacity_factor = 2
        self.image_feature_name = "image/data"
        self.caption_feature_name = "image/caption_ids"
        self.mask_feature_name ="image/mask_ids"
        #add
        self.caption_lbl_name = "image/cls_lbls"
        self.image_format = "jpeg"


        # about the model architecture
        self.cnn = 'vgg16'    # 'vgg16' or 'resnet50'
        self.dim_embedding = 512
        self.num_lstm_units = 512
        self.num_initalize_layers = 2    # 1 or 2
        self.dim_initalize_layer = 512
        self.num_attend_layers = 2       # 1 or 2
        self.dim_attend_layer = 512
        self.num_decode_layers = 2       # 1 or 2
        self.dim_decode_layer = 1024
        self.hidden_size = 512 #???
        
        # about the classification task
        self.num_classes = 512     # from instance segmentation
        self.num_attributes = 196 # 

        # about the weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.fc_kernel_regularizer_scale = 1e-4
        self.fc_activity_regularizer_scale = 0.0
        self.conv_kernel_regularizer_scale = 1e-4
        self.conv_activity_regularizer_scale = 0.0
        self.fc_drop_rate = 0.5
        self.lstm_drop_rate = 0.3
        self.attention_loss_factor = 0.01

        # about the optimization
        self.optimizer = 'Adam'    # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
        self.initial_learning_rate = 1e-5
        self.learning_rate_decay_factor = 0.99
        self.num_steps_per_decay = 10000
        self.clip_gradients = 5.0
        self.momentum = 0.0
        self.use_nesterov = True
        self.decay = 0.9
        self.centered = True
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6
        self.image_height = 224
        self.image_width = 224

        # For write shard file
        self.num_write_shard_threads = 8 
        self.shard_dir = './datasets/shard/'
        self.num_shards = 256
        self.dataset_image_dir = './datasets'
        self.eval_image_dir = './datasets/val/'
        self.test_image_dir = './datasets/val/'

        # self.train_caption_file = './datasets/rawjson/captions_train2014.json'
        # self.val_caption_file = './datasets/rawjson/captions_val2014.json'

        self.train_caption_file = './datasets/rawjson/train.json'
        self.val_caption_file = './datasets/rawjson/val.json'

        self.prepare_annotation_dir  = './datasets/prepare/'

        self.train_csv_file = './datasets/prepare/train_anns.csv'

        self.valpart_train_json_file = './datasets/prepare/valpart_train.json'
        self.valpart_eval_json_file = './datasets/prepare/valpart_eval.json'
        self.valpart_test_json_file = './datasets/prepare/valpart_test.json'
        self.word_count_file = './datasets/prepare/word_count_file.csv'

        # about the vocabulary
        self.vocabulary_file = './datasets/vocabulary.csv'

        self.results_dir = './results/'
        self.cnn_checkpoint_dir = './results/cnn/checkpoint/'
        self.cnn_checkpoint_path = './results/cnn/checkpoint/model-ckpt'
        self.checkpoint_dir = './results/caption/checkpoint/'
        self.caption_checkpoint_dir = './results/caption/checkpoint/'
        self.caption_checkpoint_path = './results/caption/checkpoint/model-ckpt'

        self.summary_dir = './summary/improved'

        self.eval_result_dir = './results/eval'
        self.eval_result_file = './results/eval/eval_results.json'
        self.save_eval_result_as_image = False

        self.test_result_dir = './results/test'
        self.test_result_file = './results/test/test_results.json'
        self.save_test_result_as_image = True

        # about the evaluation       
        
        self.eval_img_nums=10000
        self.test_img_nums=10000

        # about the testing
        self.infer_image_dir = './results/infer/images/'
        self.infer_result_dir = './results/infer/results/'
        self.infer_result_file = './results/infer/results.csv'
