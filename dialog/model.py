# coding: utf-8
import tensorflow as tf
from dialog.seq2seq import dialog_attention_seq2seq
from tensorflow.python.ops.seq2seq import sequence_loss
import numpy as np


class Seq2SeqModel(object):

    def __init__(self, vocab_size, size,
                 num_layers, max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor,
                 num_samples=512, forward_only=False, max_dialog_length = 10, max_answer_length = 20):

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        self.max_dialog_length = max_dialog_length
        self.max_answer_length = max_answer_length

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None

        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.vocab_size:
            with tf.device("/cpu:0"):
                w = tf.get_variable("proj_w", [size, self.vocab_size])
                w_t = tf.transpose(w)
                b = tf.get_variable("proj_b", [self.vocab_size])
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                with tf.device("/cpu:0"):
                    labels = tf.reshape(labels, [-1, 1])
                    return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                                                      self.vocab_size)

            softmax_loss_function = sampled_loss

        # Create the internal multi-layer cell for our RNN.
        single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        cell = single_cell
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return dialog_attention_seq2seq(
                encoder_inputs, decoder_inputs, cell, vocab_size, output_projection=output_projection,
                feed_previous=do_decode)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        for i in range(0, max_dialog_length):
            one_turn_encoder_inputs = []
            one_turn_decoder_inputs = []
            one_turn_target_weights = []
            for j in range(0, max_answer_length):
                one_turn_encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}_{1}".format(i, j)))

            for j in range(0, max_answer_length + 1):
                one_turn_decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{0}_{1}".format(i, j)))
                one_turn_target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                      name="weight{0}_{1}".format(i, j)))

            self.encoder_inputs.append(list(one_turn_encoder_inputs))
            self.decoder_inputs.append(list(one_turn_decoder_inputs))
            self.target_weights.append(list(one_turn_target_weights))

        # Our targets are decoder inputs shifted by one.
        targets = []
        for i in range(0, max_dialog_length):
            targets.append([self.decoder_inputs[i][j + 1] for j in xrange(len(self.decoder_inputs[i]) - 1)])

        # Training outputs and losses.
        if forward_only:
            self.outputs, _ = seq2seq_f(self.encoder_inputs, self.decoder_inputs, True)

            self.loss = 0
            for i in range(0, max_dialog_length):
                self.loss += sequence_loss(self.outputs[i][:-1], targets[i], self.target_weights[i][:-1],
                                        softmax_loss_function=softmax_loss_function)

            # If we use output projection, we need to project outputs for decoding.
            if output_projection is not None:
                self.outputs = tf.matmul(self.outputs, output_projection[0]) + output_projection[1]
        else:
            self.outputs, _ = seq2seq_f(self.encoder_inputs, self.decoder_inputs, False)

            self.loss = 0
            for i in range(0, max_dialog_length):
                self.loss += sequence_loss(self.outputs[i][:-1], targets[i], self.target_weights[i][:-1],
                                        softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)

            gradients = tf.gradients(self.loss, params)
            clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients,
                                                                           max_gradient_norm)
            self.update = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, forward_only):
        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}

        for i in range(0, self.max_dialog_length):
            for j in range(0, self.max_answer_length):
                input_feed[self.encoder_inputs[i][j].name] = encoder_inputs[i][j]
                input_feed[self.decoder_inputs[i][j].name] = decoder_inputs[i][j]
                input_feed[self.target_weights[i][j].name] = target_weights[i][j]


        # Since our targets are decoder inputs shifted by one, we need one more.
        for i in range(0, self.max_dialog_length):
            last_target = self.decoder_inputs[i][self.max_answer_length].name
            input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
            input_feed[self.target_weights[i][self.max_answer_length].name] \
                = target_weights[i][self.max_answer_length]


        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.update,
                           self.gradient_norm,  # Gradient norm.
                           self.loss]  # Loss for this batch.
        else:
            output_feed = [self.loss]  # Loss for this batch.
            for l in xrange(self.max_dialog_length):  # Output logits.
                output_feed.append(self.outputs[l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
          return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
          return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.
