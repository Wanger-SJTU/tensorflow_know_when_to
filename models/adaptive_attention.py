
import os
import pdb
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from .nnets import NN


class ShowAttendTell(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.train_cnn = self.is_train and config.train_cnn
        self.image_shape = [config.image_height, config.image_width, 3]
        self.nn = NN(config)

    def build(self,dataprovider=None):
        self.build_inputs(dataprovider)
        self.build_vgg16()
        self.cam_sampling()
        self.build_rnn()
        self.setup_global_step()
        if self.is_train:
            self.build_optimizer()
            self.build_summary()


    def build_inputs(self,dataprovider=None):
        if self.is_train:
            image_batch, caption_batch, mask_batch = \
                dataprovider.generate_batch_data()

            self.images = image_batch
            self.sentences = caption_batch
            self.masks = mask_batch
        else:
            self.images = tf.placeholder(dtype = tf.float32,
                            shape = [self.config.batch_size] + self.image_shape)
            self.sentences = None
            self.masks = None

    def build_vgg16(self):
        """ Build the VGG16 net. """
        config = self.config

        conv1_1_feats = self.nn.conv2d(self.images, 64, name = 'conv1_1')
        conv1_2_feats = self.nn.conv2d(conv1_1_feats, 64, name = 'conv1_2')
        pool1_feats = self.nn.max_pool2d(conv1_2_feats, name = 'pool1')

        conv2_1_feats = self.nn.conv2d(pool1_feats, 128, name = 'conv2_1')
        conv2_2_feats = self.nn.conv2d(conv2_1_feats, 128, name = 'conv2_2')
        pool2_feats = self.nn.max_pool2d(conv2_2_feats, name = 'pool2')

        conv3_1_feats = self.nn.conv2d(pool2_feats, 256, name = 'conv3_1')
        conv3_2_feats = self.nn.conv2d(conv3_1_feats, 256, name = 'conv3_2')
        conv3_3_feats = self.nn.conv2d(conv3_2_feats, 256, name = 'conv3_3')
        pool3_feats = self.nn.max_pool2d(conv3_3_feats, name = 'pool3')

        conv4_1_feats = self.nn.conv2d(pool3_feats, 512, name = 'conv4_1')
        conv4_2_feats = self.nn.conv2d(conv4_1_feats, 512, name = 'conv4_2')
        conv4_3_feats = self.nn.conv2d(conv4_2_feats, 512, name = 'conv4_3')
        pool4_feats = self.nn.max_pool2d(conv4_3_feats, name = 'pool4')

        conv5_1_feats = self.nn.conv2d(pool4_feats,   512, name = 'conv5_1')
        conv5_2_feats = self.nn.conv2d(conv5_1_feats, 512, name = 'conv5_2')
        conv5_3_feats = self.nn.conv2d(conv5_2_feats, 512, name = 'conv5_3')
        # size of conv5_3_feats is Batch*512*14*14
      
        self.conv_feats = conv5_3_feats
        self.num_ctx = 196
        self.dim_ctx = 512
   
    def build_rnn(self):
        """ Build the RNN. """
        print("Building the RNN...")
        config = self.config

        # Setup the placeholders
        if self.is_train:
            conv_feats = self.conv_feats
        else:
            conv_feats = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, self.num_ctx,  self.dim_ctx])
            last_decode_output = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units])
            last_word = tf.placeholder(
                dtype = tf.int32,
                shape = [config.batch_size])

        # Setup the word embedding
        with tf.variable_scope("word_embedding"):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocabulary_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = self.is_train)

        # Setup the LSTM
        lstm_decode = tf.nn.rnn_cell.LSTMCell(
                            config.num_lstm_units,
                initializer = self.nn.fc_kernel_initializer)
        
        lstm_encode = tf.nn.rnn_cell.LSTMCell(
                            config.num_lstm_units,
                initializer = self.nn.fc_kernel_initializer)
        
        if self.is_train:
            lstm_encode = tf.nn.rnn_cell.DropoutWrapper(
                lstm_encode,
                input_keep_prob  = 1.0-config.lstm_drop_rate,
                output_keep_prob = 1.0-config.lstm_drop_rate,
                state_keep_prob  = 1.0-config.lstm_drop_rate)
            lstm_decode = tf.nn.rnn_cell.DropoutWrapper(
                lstm_decode,
                input_keep_prob  = 1.0-config.lstm_drop_rate,
                output_keep_prob = 1.0-config.lstm_drop_rate,
                state_keep_prob  = 1.0-config.lstm_drop_rate)

        # Initialize the LSTM using the mean context
        with tf.variable_scope("initialize"):
            initial_memory, initial_output = self.initialize(global_features)
            initial_state = initial_memory, initial_output

        # Prepare to run
        predictions = []
        if self.is_train:
            alphas = []
            cross_entropies = []
            predictions_correct = []
            num_steps = config.max_caption_length
            last_output = initial_output
            last_memory = initial_memory
            last_word = tf.zeros([config.batch_size], tf.int32)
        else:
            num_steps = 1
            last_memory, last_output = last_decode_memory, last_decode_output
        
        
        last_state = last_memory, last_output

        # Generate the words one by one
        for idx in range(num_steps):

            # Embed the last word
            with tf.variable_scope("word_embedding"):
                word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                                    last_word)
            # Apply the LSTM
            with tf.variable_scope("lstm_encode"):
                # current_input = tf.concat([word_embed], 1)
                last_state = last_memory, last_output
                concat_input = tf.concat([global_features, word_embed, last_output], 1)
                encode_output, encode_state = lstm_encode(concat_input, last_state)
                encode_memory, _ = encode_state
            
            with tf.variable_scope("sentinel"):
                st = self.build_sentinal(concat_input, encode_output, encode_memory)
            
            with tf.variable_scope("attend"):
                alpha_t, beta_t, c_hat = self.attend_adaptive(spatial_feats, encode_output, st)
                if self.is_train:
                    # if train. masks the word for current training
                    tiled_masks = tf.tile(tf.expand_dims(self.masks[:, idx], 1),
                                         [1, self.num_ctx])
                    masked_alpha = alpha_t * tiled_masks
                    alphas.append(tf.reshape(masked_alpha, [-1]))

            # Decode the expanded output of LSTM into a word
            with tf.variable_scope("decode"):

                expanded_output = tf.concat([encode_output, c_hat,],
                                             axis = 1)
                decode_output, decode_state = lstm_decode(expanded_output, encode_state)
                decode_memory,_ = decode_state
                logits = self.nn.dense(decode_output, 
                                    config.vocabulary_size,
                                    use_bias=False)
                probs = tf.nn.softmax(logits)
                prediction = tf.argmax(logits, 1)
                predictions.append(prediction)
            
            # with tf.variable_scope("decode"):
            #     expanded_output = tf.concat([output,
            #                                  c_hat,
            #                                  word_embed],
            #                                  axis = 1)
            #     logits = self.decode(expanded_output)
            #     probs = tf.nn.softmax(logits)
            #     prediction = tf.argmax(logits, 1)
            #     predictions.append(prediction)

            # Compute the loss for this step, if necessary
            if self.is_train:
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels = self.sentences[:, idx],
                    logits = logits)
                masked_cross_entropy = cross_entropy * self.masks[:, idx]
                cross_entropies.append(masked_cross_entropy)

                ground_truth = tf.cast(self.sentences[:, idx], tf.int64)
                prediction_correct = tf.where(
                    tf.equal(prediction, ground_truth),
                    tf.cast(self.masks[:, idx], tf.float32),
                    tf.cast(tf.zeros_like(prediction), tf.float32))
                predictions_correct.append(prediction_correct)

                last_output = decode_output
                last_memory,_ = decode_state
                last_state = decode_state
                last_word = self.sentences[:, idx]

            tf.get_variable_scope().reuse_variables()

        # Compute the final loss, if necessary
        if self.is_train:
            cross_entropies = tf.stack(cross_entropies, axis = 1)
            cross_entropy_loss = tf.reduce_sum(cross_entropies) \
                                 / tf.reduce_sum(self.masks)

            alphas = tf.stack(alphas, axis = 1)
            alphas = tf.reshape(alphas, [config.batch_size, self.num_ctx, -1])
            attentions = tf.reduce_sum(alphas, axis = 2)
            diffs = tf.ones_like(attentions) - attentions
            attention_loss = config.attention_loss_factor \
                             * tf.nn.l2_loss(diffs) \
                             / (config.batch_size * self.num_ctx)

            reg_loss = tf.losses.get_regularization_loss()

            total_loss = cross_entropy_loss + attention_loss \
                         + reg_loss+ 0.5*self.loss_op

            predictions_correct = tf.stack(predictions_correct, axis = 1)
            accuracy = tf.reduce_sum(predictions_correct) \
                       / tf.reduce_sum(self.masks)

        
        if self.is_train:
            # self.contexts = contexts
            self.channel_feats = channel_feats
            self.spatial_feats = spatial_feats
            self.global_feats  = global_features
            self.total_loss = total_loss
            self.cross_entropy_loss = cross_entropy_loss
            self.attention_loss = attention_loss
            self.reg_loss = reg_loss
            self.accuracy = accuracy
            self.attentions = attentions
        else:
            # self.contexts = contexts
            self.channel_feats = channel_feats
            self.spatial_feats = spatial_feats
            self.global_feats  = global_features

            self.initial_memory = initial_memory
            self.initial_output = initial_output
            self.last_memory = last_memory
            self.last_output = last_output
            self.last_word = last_word
            self.memory = decode_memory
            self.output = decode_output
            self.probs = probs
            self.alpha = alpha_t # for geting the alpha

        print("RNN built.")

    def initialize(self, context_mean):
        """ Initialize the LSTM using the mean context. """
        config = self.config
        context_mean = self.nn.dropout(context_mean)
        if config.num_initalize_layers == 1:
            # use 1 fc layer to initialize
            memory = self.nn.dense(context_mean,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_a')
            output = self.nn.dense(context_mean,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_b')
        else:
            # use 2 fc layers to initialize
            temp1 = self.nn.dense(context_mean,
                                  units = config.dim_initalize_layer,
                                  activation = tf.tanh,
                                  name = 'fc_a1')
            temp1 = self.nn.dropout(temp1)
            memory = self.nn.dense(temp1,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_a2')

            temp2 = self.nn.dense(context_mean,
                                  units = config.dim_initalize_layer,
                                  activation = tf.tanh,
                                  name = 'fc_b1')
            temp2 = self.nn.dropout(temp2)
            output = self.nn.dense(temp2,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_b2')
        return memory, output

    def build_sentinal(self, concat_input, decoder_hidden, memory_cell):
        """
        concat_input: 
            concatenation of the word embedding vector and 
            the global image features retuend by the encoder.
            The word embedding shape: (batch_size, embed_size) 
            global_image shape      : (batch_size, embed_size)
            concat_input of shape   : (batch_size, embed * 2)
        
        decoder_hidden: hidden state of the decoder, 
                        (batch_size, hidden_size)
        memory_cell: memory state of the decoder, 
                     (batch_size, hidden_size)
        """
        config = self.config
        if self.is_train:
            decoder_hidden = self.nn.dropout(decoder_hidden)
            concat_input   = self.nn.dropout(concat_input)
        
        sen_hidd = self.nn.dense(decoder_hidden,
                                units=config.hidden_size,
                                activation = None,
                                use_bias = False,
                                    )
        sen_input = self.nn.dense(concat_input,
                                units=config.hidden_size,
                                activation = None,
                                use_bias = False,
                                )
        gt = tf.nn.sigmoid(sen_hidd + sen_input)# (batch_size, hidden_size)
        st = gt * tf.nn.tanh(memory_cell)# (batch_size, hidden_size)
        return st

    def attend_adaptive(self, spatial_features, output, st):
        config = self.config
        
        if self.is_train:
            spatial_features = self.nn.dropout(spatial_features)
            output = self.nn.dropout(output)
        # spatial_features
        #  -batch * 196 * hidden_size(512)
        cnn_out = self.nn.dense(spatial_features,
                                config.dim_attend_layer,
                                use_bias=False)
        #  -batch * 196 * dim_attend_size(512)

        # batch*dim_attend_layer
        dec_out = self.nn.dense(output,
                                config.dim_attend_layer,
                                use_bias=False)
        # batch*dim_attend_layer
        
        addition_out = tf.nn.tanh(cnn_out + tf.expand_dims(dec_out,1))
        # batch * 196 * dim_attend_size(512)
        if self.is_train:
            addition_out= self.nn.dropout(addition_out)
        # batch * 196
        zt = self.nn.dense(addition_out, 1, use_bias=False)
        zt = tf.squeeze(zt, 2)
        alpha_t = tf.nn.softmax(zt)
        ct = tf.reduce_sum(spatial_features*tf.expand_dims(alpha_t,2), axis=1)

        out = tf.nn.tanh(self.nn.dense(
                            (self.nn.dense(st,
                            config.dim_attend_layer,
                            use_bias=False) +output),
                            config.dim_attend_layer,
                            use_bias=False))

        out = self.nn.dense(out, 1, use_bias=False)
        concat = tf.concat([zt, out], axis=1)   
        alpha_hat = tf.nn.softmax(concat)
        beta_t = tf.expand_dims(alpha_hat[:,-1], 1)
        c_hat = beta_t * st + (1 - beta_t) * ct
        return alpha_t, beta_t, c_hat

    def attend(self, contexts, output):
        """ Attention Mechanism. """
        config = self.config
        reshaped_contexts = tf.reshape(contexts, [-1, self.dim_ctx])
        reshaped_contexts = self.nn.dropout(reshaped_contexts)
        output = self.nn.dropout(output)
        if config.num_attend_layers == 1:
            # use 1 fc layer to attend
            logits1 = self.nn.dense(reshaped_contexts,
                                    units = 1,
                                    activation = None,
                                    use_bias = False,
                                    name = 'fc_a')
            logits1 = tf.reshape(logits1, [-1, self.num_ctx])
            logits2 = self.nn.dense(output,
                                    units = self.num_ctx,
                                    activation = None,
                                    use_bias = False,
                                    name = 'fc_b')
            logits = logits1 + logits2
        else:
            # use 2 fc layers to attend
            temp1 = self.nn.dense(reshaped_contexts,
                                  units = config.dim_attend_layer,
                                  activation = tf.tanh,
                                  name = 'fc_1a')
            temp2 = self.nn.dense(output,
                                  units = config.dim_attend_layer,
                                  activation = tf.tanh,
                                  name = 'fc_1b')
            temp2 = tf.tile(tf.expand_dims(temp2, 1), [1, self.num_ctx, 1])
            temp2 = tf.reshape(temp2, [-1, config.dim_attend_layer])
            temp = temp1 + temp2
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = 1,
                                   activation = None,
                                   use_bias = False,
                                   name = 'fc_2')
            logits = tf.reshape(logits, [-1, self.num_ctx])
        alpha = tf.nn.softmax(logits)
        return alpha

    def decode(self, expanded_output):
        """ Decode the expanded output of the LSTM into a word. """
        config = self.config
        expanded_output = self.nn.dropout(expanded_output)
        if config.num_decode_layers == 1:
            # use 1 fc layer to decode
            logits = self.nn.dense(expanded_output,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc')
        else:
            # use 2 fc layers to decode
            temp = self.nn.dense(expanded_output,
                                 units = config.dim_decode_layer,
                                 activation = tf.tanh,
                                 name = 'fc_1')
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc_2')
        return logits

    def build_optimizer(self):
        """ Setup the optimizer and training operation. """
        config = self.config

        learning_rate = tf.constant(config.initial_learning_rate)
        if config.learning_rate_decay_factor < 1.0:
            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps = config.num_steps_per_decay,
                    decay_rate = config.learning_rate_decay_factor,
                    staircase = True)
            learning_rate_decay_fn = _learning_rate_decay_fn
        else:
            learning_rate_decay_fn = None

        with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE):
            if config.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(
                    learning_rate = config.initial_learning_rate,
                    beta1 = config.beta1,
                    beta2 = config.beta2,
                    epsilon = config.epsilon
                    )
            elif config.optimizer == 'RMSProp':
                optimizer = tf.train.RMSPropOptimizer(
                    learning_rate = config.initial_learning_rate,
                    decay = config.decay,
                    momentum = config.momentum,
                    centered = config.centered,
                    epsilon = config.epsilon
                )
            elif config.optimizer == 'Momentum':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate = config.initial_learning_rate,
                    momentum = config.momentum,
                    use_nesterov = config.use_nesterov
                )
            else:
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate = config.initial_learning_rate
                )

            opt_op = tf.contrib.layers.optimize_loss(
                loss = self.total_loss,
                global_step = self.global_step,
                learning_rate = learning_rate,
                optimizer = optimizer,
                clip_gradients = config.clip_gradients,
                learning_rate_decay_fn = learning_rate_decay_fn)

        self.opt_op = opt_op

    def build_summary(self):
        """ Build the summary (for TensorBoard visualization). """
        with tf.name_scope("variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)

        with tf.name_scope("metrics"):
            tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
            tf.summary.scalar("attention_loss", self.attention_loss)
            tf.summary.scalar("reg_loss", self.reg_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("accuracy", self.accuracy)
            tf.summary.scalar("cls loss", self.loss_op)

        with tf.name_scope("attentions"):
            self.variable_summary(self.attentions)

        self.summary = tf.summary.merge_all()

    def variable_summary(self, var):
        """ Build the summary for a variable. """
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    def load_cnn(self, session, data_path, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading the CNN from %s..." %data_path)
        data_dict = np.load(data_path).item()
        count = 0
        for op_name in tqdm(data_dict, ncols=60):
            with tf.variable_scope(op_name, reuse = True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        count += 1
                    except ValueError:
                        pass
        print("%d tensors loaded." %count)

    def setup_graph_from_checkpoint(self,sess,checkpoint_dir):
        if tf.gfile.IsDirectory(checkpoint_dir):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
            if not checkpoint_path:
                raise ValueError("No checkpoint file found in: %s" % checkpoint_path)
        file_name = checkpoint_path.split('/')[-1]
        saver = tf.train.Saver()
        tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
        saver.restore(sess, checkpoint_path)
        tf.logging.info("Successfully loaded checkpoint: %s",
                            os.path.basename(checkpoint_path))
        print(int(file_name.split('-')[-1]))
        return int(file_name.split('-')[-1])

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
                initial_value=0,
                name="global_step",
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step
