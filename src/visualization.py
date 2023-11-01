# functions that create plots
import json
import os.path

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from src.data_processing import get_n_layers
from src.utils import makedirs


def get_absmax_clim(nd_array):
    a = np.max(np.abs(nd_array))
    return np.array([-a, a])


def compute_interpolated_activation_values(topomap_data_dir, layer, resolution=100):
    layer_id = "layer" + str(layer).zfill(3)

    coordinates_path = os.path.join(topomap_data_dir, layer_id + "_coordinates.npy")
    if os.path.isfile(coordinates_path):
        coordinates = np.load(coordinates_path)

        coordinates = coordinates[:, ::-1]  # to account for the transpose in the interpolation imshow
        coordinates[:, 0] = 1 - coordinates[:, 0]  # to account for the invert_y in the interpolation imshow

        xx_min, yy_min = np.min(coordinates, axis=0)
        xx_max, yy_max = np.max(coordinates, axis=0)
        resolution = 100

        xx, yy = np.mgrid[xx_min:xx_max:complex(resolution), yy_min:yy_max:complex(resolution)]

        topomap_activations_path = os.path.join(topomap_data_dir, layer_id + "_activations.npy")
        topomap_activations = np.load(topomap_activations_path)

        interpolated_activations = np.zeros(shape=[topomap_activations.shape[0], resolution, resolution])

        for group_id, group_activations in enumerate(topomap_activations):
            new_xy = interpolate.griddata(coordinates,
                                          group_activations,
                                          (xx, yy),
                                          method='linear')
            new_xy[np.isnan(new_xy)] = 0

            interpolated_activations[group_id] = np.transpose(new_xy[:, ::-1])

        interpolated_activations_path = os.path.join(topomap_data_dir, layer_id + "_interpolated_activations.npy")
        np.save(interpolated_activations_path, interpolated_activations)


def plot_layer_topomaps_individually(topomap_data_dir, plot_output_dir, layer, group_names):
    layer_id = "layer" + str(layer).zfill(3)

    coordinates_path = os.path.join(topomap_data_dir, layer_id + "_coordinates.npy")
    if os.path.isfile(coordinates_path):
        layer_plot_output_dir = os.path.join(plot_output_dir, layer_id)
        makedirs([layer_plot_output_dir])

        interpolated_activations_path = os.path.join(topomap_data_dir, layer_id + "_interpolated_activations.npy")
        interpolated_activations = np.load(interpolated_activations_path)

        clim = get_absmax_clim(interpolated_activations)

        for group_id, group_name in enumerate(group_names):
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(interpolated_activations[group_id],
                      clim=clim,
                      cmap='bwr')
            # ax.set_title(group_name)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("white")

            topomap_plot_path = os.path.join(layer_plot_output_dir, group_name + '.pdf')
            plt.savefig(topomap_plot_path, bbox_inches='tight')
            plt.close(fig)


def plot_layer_topomaps_jointly(topomap_data_dir, plot_output_dir, layer):
    layer_id = "layer" + str(layer).zfill(3)

    coordinates_path = os.path.join(topomap_data_dir, layer_id + "_coordinates.npy")
    if os.path.isfile(coordinates_path):
        layer_plot_output_dir = os.path.join(plot_output_dir, layer_id)
        makedirs([layer_plot_output_dir])

        interpolated_activations_path = os.path.join(topomap_data_dir, layer_id + "_interpolated_activations.npy")
        interpolated_activations = np.load(interpolated_activations_path)

        clim = get_absmax_clim(interpolated_activations)

        fig, ax = plt.subplots(1, 1, figsize=(5*interpolated_activations.shape[0], 5))

        interpolated_activations = np.concatenate(interpolated_activations,1)

        ax.imshow(interpolated_activations,
                  clim=clim,
                  cmap='bwr')
        # ax.set_title(group_name)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")

        topomap_plot_path = os.path.join(layer_plot_output_dir, 'overview.pdf')
        plt.savefig(topomap_plot_path, bbox_inches='tight')
        plt.close(fig)


def plot_topomaps(processed_corpus_path, mode="all_in_row"):
    topomap_data_dir = os.path.join(processed_corpus_path, "topomap_data")

    n_layers = get_n_layers(processed_corpus_path)
    group_names = np.load(os.path.join(processed_corpus_path, "contrastive_naps", "group_names.npy"))

    plot_output_dir = os.path.join(processed_corpus_path, "topomap_plots")
    makedirs([plot_output_dir])

    for layer in range(n_layers - 1):
        compute_interpolated_activation_values(topomap_data_dir, layer)
        if mode == "individually":
            plot_layer_topomaps_individually(topomap_data_dir, plot_output_dir, layer, group_names)
        elif mode == "all_in_row":
            plot_layer_topomaps_jointly(topomap_data_dir, plot_output_dir, layer)
        else:
            print(mode, ": mode unknown.")

    return None
