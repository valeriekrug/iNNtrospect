# script for performing iNNtrospection
# fixed instances of this script go to examples
import json

from env_setup import local_env_settings
from src.naps import compute_contrastive_naps
from src.topomaps import compute_topomap_layout

local_env_settings()

config_file = "../configs/MNIST_CNN_SHALLOW.json"
with open(config_file, "r") as f:
    config = json.load(f)


group_names_of_interest = None#["0","2","4","9"]
weight_by_group_size = False

compute_contrastive_naps(config["processed_corpus_path"],
                         group_names_of_interest = group_names_of_interest,
                         weight_by_group_size = weight_by_group_size)

compute_topomap_layout(config["processed_corpus_path"],
                       layouting_method="tSNE",
                       distribute_in_circle=True,
                       from_contrastive_naps=True)


# TODO create visualization