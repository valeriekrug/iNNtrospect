# functions to compute Neuron Activation Profiles
import json
import os

from tqdm import tqdm

from constants.check_constants import PIPELINE_STEPS
from constants.directory_constants import OUTPUT_DIRECTORY_NAMES
from src.checks import check_pipeline_dependencies, check_group_names_of_interest
from src.data_processing import get_n_layers
from src.representation_analysis import group_names_to_indices
from src.utils import makedirs
import numpy as np

def compute_group_sizes(processed_corpus_path):
    layer_id = "layer000"
    activations_dir = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.ACTS, layer_id)

    with open(os.path.join(processed_corpus_path, "group_name_to_files.json"), "r") as f:
        group_to_file_dict = json.load(f)
    groups = [*group_to_file_dict.keys()]

    group_sizes = list()
    for group in tqdm(groups, desc="compute group sizes"):
        group_files = group_to_file_dict[group]

        n_examples_per_batch = list()
        for batch_file in group_files:
            batch_path = os.path.join(activations_dir, batch_file)

            batch = np.load(batch_path)
            n_examples_per_batch.append(batch.shape[0])

        n_examples_in_group = np.sum(n_examples_per_batch)
        group_sizes.append(n_examples_in_group)


    group_size_file_path = os.path.join(processed_corpus_path, "group_sizes.npy")
    np.save(group_size_file_path, np.array(group_sizes))

def compute_group_weights(processed_corpus_path, indices_of_interest, weight_by_group_size):
    group_weights = np.ones(shape=len(indices_of_interest))
    if weight_by_group_size:
        group_weights = np.load(os.path.join(processed_corpus_path, "group_sizes.npy"))
        group_weights = group_weights[indices_of_interest]
    group_weights = group_weights / np.sum(group_weights)
    return group_weights



def compute_group_average(processed_corpus_path, layer_id, acts_from_aligned, group_files):

    if acts_from_aligned:
        activations_dir = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.ALIGNED, layer_id)
    else:
        activations_dir = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.ACTS, layer_id)

    n_examples_per_batch = list()
    batch_averages = list()
    for batch_file in group_files:
        batch_path = os.path.join(activations_dir, batch_file)

        batch = np.load(batch_path)
        n_examples_per_batch.append(batch.shape[0])
        batch_averages.append(np.mean(batch, 0))

    group_average = np.zeros_like(batch_averages[0])
    n_examples_in_group = np.sum(n_examples_per_batch)
    frac_examples_per_batch = n_examples_per_batch / n_examples_in_group
    for batch_average, p_batch_examples in zip(batch_averages, frac_examples_per_batch):
        group_average = group_average + (batch_average * p_batch_examples)

    return group_average

def compute_and_save_layer_nap(processed_corpus_path, layer, nap_output_dir, use_aligned_acts):
    acts_from_aligned = False
    layer_id = "layer" + str(layer).zfill(3)
    test_activation_dir = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.ALIGNED, layer_id)
    if use_aligned_acts and os.path.isdir(test_activation_dir):
        acts_from_aligned = True

    with open(os.path.join(processed_corpus_path, "group_name_to_files.json"), "r") as f:
        group_to_file_dict = json.load(f)
    groups = [*group_to_file_dict.keys()]

    layer_naps = list()
    for group in tqdm(groups,desc="averaging groups"):
        group_files = group_to_file_dict[group]
        group_average = compute_group_average(processed_corpus_path, layer_id, acts_from_aligned, group_files)
        layer_naps.append(group_average)
    layer_naps = np.array(layer_naps)

    nap_file_path = os.path.join(nap_output_dir, layer_id + ".npy")
    np.save(nap_file_path, layer_naps)

def compute_naps(processed_corpus_path, use_aligned_acts=True):
    if use_aligned_acts:
        check_pipeline_dependencies(processed_corpus_path, PIPELINE_STEPS.NAPS_ALIGNED)
    else:
        check_pipeline_dependencies(processed_corpus_path, PIPELINE_STEPS.NAPS)

    n_layers = get_n_layers(processed_corpus_path)

    nap_output_dir = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.NAPS)
    makedirs([nap_output_dir])

    compute_group_sizes(processed_corpus_path)

    for layer in range(n_layers - 1):
        compute_and_save_layer_nap(processed_corpus_path, layer, nap_output_dir, use_aligned_acts)


def compute_and_save_contrastive_layer_nap(nap_dir, output_dir, layer, indices_of_interest, group_weights, drop_other_groups=True):


    layer_id = "layer" + str(layer).zfill(3)
    nap_path = os.path.join(nap_dir, layer_id + ".npy")

    nap = np.load(nap_path)
    nap_subset = nap[indices_of_interest]

    dims_to_append = list(np.arange(len(nap_subset.shape) - 1) + 1)
    group_weights = np.expand_dims(group_weights, dims_to_append)
    global_average = np.sum(nap_subset * group_weights, 0)

    if drop_other_groups:
        nap = nap_subset - np.expand_dims(global_average,0)
    else:
        nap = nap - np.expand_dims(global_average, 0)

    nap_file_path = os.path.join(output_dir, layer_id + ".npy")
    np.save(nap_file_path, nap)

def compute_contrastive_naps(processed_corpus_path, group_names_of_interest=None, weight_by_group_size=False, drop_other_groups=True):
    check_pipeline_dependencies(processed_corpus_path, PIPELINE_STEPS.CONTRASTIVE_NAPS)

    contrastive_nap_group_names, indices_of_interest = group_names_to_indices(processed_corpus_path, group_names_of_interest)

    n_layers = get_n_layers(processed_corpus_path)

    group_weights = compute_group_weights(processed_corpus_path, indices_of_interest, weight_by_group_size)

    nap_output_dir = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.NAPS)
    contrastive_nap_output_dir = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.CONTRASTIVE_NAPS)
    makedirs([contrastive_nap_output_dir])

    if drop_other_groups:
        np.save(os.path.join(contrastive_nap_output_dir, "group_names.npy"), contrastive_nap_group_names)
    else:
        with open(os.path.join(processed_corpus_path, "group_name_to_index.json"), "r") as f:
            group_name_to_index = json.load(f)
        group_names = np.array([*group_name_to_index.keys()])
        np.save(os.path.join(contrastive_nap_output_dir, "group_names.npy"), group_names)

    for layer in range(n_layers - 1):
        compute_and_save_contrastive_layer_nap(nap_output_dir,
                                               contrastive_nap_output_dir,
                                               layer,
                                               indices_of_interest,
                                               group_weights,
                                               drop_other_groups)