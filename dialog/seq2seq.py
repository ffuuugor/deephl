# coding: utf-8

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.seq2seq import embedding_attention_decoder


def dialog_attention_seq2seq(encoder_inputs, decoder_inputs, cell, vocab_size,
                             num_heads=1, output_projection=None,
                             feed_previous=False, dtype=dtypes.float32,
                             scope=None, initial_state_attention=False):
    if len(encoder_inputs) != len(decoder_inputs):
        raise Exception

    with variable_scope.variable_scope(scope or "dialog_attention_seq2seq"):

        encoder_cell = rnn_cell.EmbeddingWrapper(cell, vocab_size)
        outputs = []

        fixed_batch_size = encoder_inputs[0][0].get_shape().with_rank_at_least(1)[0]
        if fixed_batch_size.value:
          batch_size = fixed_batch_size.value
        else:
          batch_size = array_ops.shape(encoder_inputs[0][0])[0]

        drnn_state = cell.zero_state(batch_size, dtype)

        for i in range(0, len(encoder_inputs)):
            if i > 0: variable_scope.get_variable_scope().reuse_variables()

            encoder_outputs, encoder_state = rnn.rnn(
                encoder_cell, encoder_inputs[i], dtype=dtype)

            # First calculate a concatenation of encoder outputs to put attention on.
            top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                          for e in encoder_outputs]
            attention_states = array_ops.concat(1, top_states)

            with variable_scope.variable_scope("DRNN"):
                drnn_out, drnn_state = cell(encoder_state, drnn_state)

            # Decoder.
            output_size = None
            if output_projection is None:
                cell = rnn_cell.OutputProjectionWrapper(cell, vocab_size)
                output_size = vocab_size

            answer_output, answer_state = embedding_attention_decoder(
                decoder_inputs[i], drnn_state, attention_states, cell,
                vocab_size, num_heads=num_heads, output_size=output_size,
                output_projection=output_projection, feed_previous=feed_previous,
                initial_state_attention=initial_state_attention)

            outputs.append(answer_output)
            with variable_scope.variable_scope("DRNN", reuse=True):
                drnn_out, drnn_state = cell(answer_state, drnn_state)

        return outputs, drnn_state

