import os
import tensorflow as tf
import numpy as np
from src.utils import makedirs


def create_acts_grads_output_dirs(output_path, n_layers):
    output_acts_path = os.path.join(output_path, "acts")
    output_grads_path = os.path.join(output_path, "grads")
    for i in range(n_layers):
        layer_running_id = 'layer' + str(i).zfill(3)
        layer_output_acts_path = os.path.join(output_acts_path, layer_running_id)
        layer_output_grads_path = os.path.join(output_grads_path, layer_running_id)
        makedirs([layer_output_acts_path,
                  layer_output_grads_path])

def get_acts_and_grads_of_batch(model, batch):
    batch = tf.convert_to_tensor(batch)

    with tf.GradientTape() as tape:
        tape.watch(batch)
        batch_activations = model(batch)

        outputs = batch_activations[-1]
        preds = np.argmax(outputs, 1)
        pred_logits = [o[p].numpy() for o, p in zip(outputs, preds)]

        batch_grads = tape.gradient(batch_activations[-1],
                                    batch_activations[:-1],
                                    output_gradients=tf.one_hot(preds, outputs.shape[-1]))
    return batch_activations, batch_grads

def save_batch_acts_and_grads_per_layer(acts, grads, batch_file, output_path):
    output_acts_path = os.path.join(output_path, "acts")
    output_grads_path = os.path.join(output_path, "grads")
    for layer_of_interest_id, (layer_acts, layer_grads) in enumerate(zip(acts, grads)):
        np.save(os.path.join(output_acts_path, 'layer' + str(layer_of_interest_id).zfill(3), batch_file),
                layer_acts)
        np.save(os.path.join(output_grads_path, 'layer' + str(layer_of_interest_id).zfill(3), batch_file),
                layer_grads)

def process_corpus_file(data_path, model, output_path):

    create_acts_grads_output_dirs(output_path, len(model.outputs))

    corpus_path = os.path.join(data_path, "corpus.csv")
    corpus = np.loadtxt(corpus_path, dtype=str, delimiter=',')
    group_names = np.unique(corpus[:, 1])

    for batch_info in corpus:
        batch_file = batch_info[0]
        group_name = batch_info[1]
        label = None
        if len(batch_info) > 2:
            label = int(batch_info[2])

        batch = np.load(os.path.join(data_path, batch_file))
        batch_activations, batch_grads = get_acts_and_grads_of_batch(model,batch)
        save_batch_acts_and_grads_per_layer(batch_activations, batch_grads, batch_file, output_path)

