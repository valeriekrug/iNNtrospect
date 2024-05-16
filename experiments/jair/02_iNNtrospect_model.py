import os
import sys

sys.path.insert(0, ".")


import json

from constants.topomap_constants import TOPOMAP_METHODS, TOPOMAP_PLOTTING_MODES
from src.data_processing import align_data
from src.model import load_imagenet_pretrained_model, create_model_with_layer_outputs
from src.visualization import plot_topomaps
from experiments.jair.specific_function import process_intersectional_corpus_file, plot_topomaps_intersectionally, \
    representation_cluster_plots, representation_cluster_bias_plots
from src.naps import compute_contrastive_naps, compute_naps
from src.topomaps import compute_topomap_layout, compute_topomap_activations


config_file = "experiments/jair/fairface_VGG16.json"
with open(config_file, "r") as f:
    config = json.load(f)

# skip_long_compute  = True
#
# if not skip_long_compute:
model, preprocessing_fun = load_imagenet_pretrained_model(config["model_name"])

# make model with output layers of interest
model_with_layer_outputs = create_model_with_layer_outputs(model,
                                                           config["layer_names_of_interest"])

# computing activations and gradients
# saved to <processed_corpus_path>/acts/ and <processed_corpus_path>/grads/
process_intersectional_corpus_file(config["data_path"],
                                   model_with_layer_outputs,
                                   preprocessing_fun,
                                   config["processed_corpus_path"])

# creating aligned activations in every layer where possible
# saved to <processed_corpus_path>/aligned/
if config["use_aligned_naps"]:
    align_data(config["processed_corpus_path"])

# compute group averages in each layer
# saved to <processed_corpus_path>/naps/
compute_naps(config["processed_corpus_path"], use_aligned_acts=config["use_aligned_naps"])

with open(os.path.join(config["processed_corpus_path"], "group_name_to_index.json"), "r") as f:
    group_to_index_dict = json.load(f)
group_names = [*group_to_index_dict.keys()]
group_names_of_interest_naps = [g for g in group_names if len(g.split("#"))==3]
compute_contrastive_naps(config["processed_corpus_path"],
                         group_names_of_interest = group_names_of_interest_naps,
                         weight_by_group_size = False,
                         drop_other_groups = False)

compute_topomap_layout(config["processed_corpus_path"],
                       layouting_method = TOPOMAP_METHODS.UMAP,
                       distribute_in_circle = True,
                       from_contrastive_naps = True,
                       group_subset = group_names_of_interest_naps)

compute_topomap_activations(config["processed_corpus_path"],
                            from_contrastive_naps = True)

plot_topomaps(config["processed_corpus_path"],
              mode=TOPOMAP_PLOTTING_MODES.SINGLE)
plot_topomaps_intersectionally("experiments/jair/",config["processed_corpus_path"])

representation_cluster_plots("experiments/jair/",config["processed_corpus_path"])

representation_cluster_bias_plots(config["processed_corpus_path"])