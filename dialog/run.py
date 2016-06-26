# coding: utf-8
import tensorflow as tf
from dialog.model import Seq2SeqModel
import time
from dialog.utils import gen_batches
import math
import os
import sys

VOCAB_SIZE=20000
NUM_UNITS = 1024
NUM_LAYERS = 2
MAX_GRADIENT_NORM = 5.0
BATCH_SIZE=32
LEARNING_RATE=0.5
DECAY_FACTOR=0.99
NUM_SAMPLES=512
MAX_DIALOG_LENGTH=10
MAX_ANSWER_LENGTH=20
STEPS_PER_CHECKPOINT=200

def main():
    with tf.Session() as session:
        model = Seq2SeqModel(vocab_size=VOCAB_SIZE,
                             size=NUM_UNITS,
                             num_layers=NUM_LAYERS,
                             max_gradient_norm=MAX_GRADIENT_NORM,
                             batch_size=BATCH_SIZE,
                             learning_rate=LEARNING_RATE,
                             learning_rate_decay_factor=DECAY_FACTOR,
                             num_samples=NUM_SAMPLES,
                             max_dialog_length=MAX_DIALOG_LENGTH,
                             max_answer_length=MAX_ANSWER_LENGTH
                             )

        session.run(tf.initialize_all_variables())
        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        for batch in gen_batches(batch_size=BATCH_SIZE,
                                 answer_size=MAX_ANSWER_LENGTH,
                                 dialog_size=MAX_DIALOG_LENGTH*2):

            start_time = time.time()
            encoder_inputs, decoder_inputs, weights = batch
            _, step_loss, _ = model.step(session, encoder_inputs, decoder_inputs, weights, False)
            step_time += (time.time() - start_time) / STEPS_PER_CHECKPOINT
            loss += step_loss / STEPS_PER_CHECKPOINT
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % STEPS_PER_CHECKPOINT == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))

                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                  session.run(model.learning_rate_decay_op)

                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join("/tmp", "dialog.ckpt")
                model.saver.save(session, checkpoint_path, global_step=model.global_step)

                step_time, loss = 0.0, 0.0
                sys.stdout.flush()

if __name__ == '__main__':
    main()

