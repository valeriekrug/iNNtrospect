# script for performing iNNtrospection
# fixed instances of this script go to examples
import json

from env_setup import local_env_settings
from src.interaction import get_options_in_preprocessing_path
from src.naps import compute_contrastive_naps

local_env_settings()

config_file = "../configs/MNIST_CNN_SHALLOW.json"
with open(config_file, "r") as f:
    config = json.load(f)


group_names_of_interest = None#["0","2","4","9"]
weight_by_group_size = False

compute_contrastive_naps(config["processed_corpus_path"],
                         group_names_of_interest = group_names_of_interest,
                         weight_by_group_size = weight_by_group_size)

# TODO compute topomap layouts
# TODO create visualization