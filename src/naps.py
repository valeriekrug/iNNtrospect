# functions to compute Neuron Activation Profiles
import json
import os

from src.data_processing import get_n_layers
from src.utils import makedirs
import numpy as np


def compute_group_average(processed_corpus_path, layer_id, acts_from_aligned, group_files):

    n_examples_per_batch = list()
    batch_averages = list()
    for batch_file in group_files:
        if acts_from_aligned:
            activations_dir = os.path.join(processed_corpus_path, "aligned", layer_id)
        else:
            activations_dir = os.path.join(processed_corpus_path, "acts", layer_id)
        batch_path = os.path.join(activations_dir, batch_file)

        batch = np.load(batch_path)
        n_examples_per_batch.append(batch.shape[0])
        batch_averages.append(np.mean(batch, 0))

    group_average = np.zeros_like(batch_averages[0])
    frac_examples_per_batch = n_examples_per_batch / np.sum(n_examples_per_batch)
    for batch_average, p_batch_examples in zip(batch_averages, frac_examples_per_batch):
        group_average = group_average + (batch_average * p_batch_examples)

    return group_average

def compute_and_save_layer_nap(processed_corpus_path, layer, nap_output_dir):
    acts_from_aligned = False
    layer_id = "layer" + str(layer).zfill(3)
    test_activation_dir = os.path.join(processed_corpus_path, "aligned", layer_id)
    if os.path.isdir(test_activation_dir):
        acts_from_aligned = True

    with open(os.path.join(processed_corpus_path, "group_to_file.json"), "r") as f:
        group_to_file_dict = json.load(f)
    groups = [*group_to_file_dict.keys()]

    layer_naps = list()
    for group in groups:
        group_files = group_to_file_dict[group]
        group_average = compute_group_average(processed_corpus_path, layer_id, acts_from_aligned, group_files)
        layer_naps.append(group_average)
    layer_naps = np.array(layer_naps)

    nap_file_path = os.path.join(nap_output_dir, layer_id + ".npy")
    np.save(nap_file_path, layer_naps)

def compute_naps(processed_corpus_path):
    n_layers = get_n_layers(processed_corpus_path)

    nap_output_dir = os.path.join(processed_corpus_path, "naps")
    makedirs([nap_output_dir])

    for layer in range(n_layers - 1):
        compute_and_save_layer_nap(processed_corpus_path, layer, nap_output_dir)



