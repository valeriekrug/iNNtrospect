# interactive notebook for running a pipeline with interactive prompts on command line
# fixed instances of this script go to examples

from env_setup import ignore_user_installs, set_active_GPU
from src.data_processing import process_corpus_file, align_data

from src.model import load_model, create_model_with_layer_outputs

import os
import numpy as np

ignore_user_installs('akrug')
set_active_GPU(0)

# "config" dummies
# model_path = '/data/project/ANNalyzer/models/MNIST_MLP'
# layer_names_of_interest = ['layer_2']
# data_path = '/data/project/iNNtrospect/MNIST/'
# processed_corpus_path = '../output/MNIST_MLP'


model_path = '/data/project/ANNalyzer/models/MNIST_CNN_SHALLOW'
layer_names_of_interest = ['conv2d_24','conv2d_25','flatten_8']
data_path = '/data/project/iNNtrospect/MNIST/'
processed_corpus_path = '../output/MNIST_CNN'

# load model
model = load_model(model_path)

# make model with output layers of interest
model_with_layer_outputs = create_model_with_layer_outputs(model, layer_names_of_interest)

# provide (batches) of examples with group annotation
# format:
# filename, groupname, (expected output)
# batch1.npy, shirt, (0)
# batch2.npy, shirt, (0)
# batch3.npy, trouser, (1)

process_corpus_file(data_path,
                    model_with_layer_outputs,
                    processed_corpus_path)

align_data(processed_corpus_path)

# TODO compute NAPs
# TODO compute topomap layouts
# TODO create visualization
