# interactive notebook for running a pipeline with interactive prompts on command line
# fixed instances of this script go to examples

from env_setup import ignore_user_installs, set_active_GPU
from src.data_processing import process_corpus_file

from src.model import load_model, create_model_with_layer_outputs

import os
import numpy as np

ignore_user_installs('akrug')
set_active_GPU(0)


# load model
model_path = '/data/project/ANNalyzer/models/MNIST_MLP'
model = load_model(model_path)

# ask for layers of interest
layer_names_of_interest = ['layer_2']

model_with_layer_outputs = create_model_with_layer_outputs(model, layer_names_of_interest)

# provide (batches) of examples with group annotation
# format:
# filename, groupname, expected output
# batch1.npy, shirt, (0)
# batch2.npy, shirt, (0)
# batch3.npy, trouser, (1)
data_path = '/data/project/iNNtrospect/MNIST/'
output_path = '../output/MNIST'

process_corpus_file(data_path,
                    model_with_layer_outputs,
                    output_path)

# TODO add saliency-alignment of preprocessed corpus

# TODO compute NAPs
# TODO compute topomap layouts
# TODO create visualization
