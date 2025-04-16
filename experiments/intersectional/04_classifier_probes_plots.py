import os
import sys

sys.path.insert(0, ".")


import pickle
import matplotlib.pyplot as plt
import numpy as np

# set directory
probe_experiment_dir = "probes/intersectional/"

dir_files = os.listdir(probe_experiment_dir)
for model_name in ["VGG16", "ResNet50", "InceptionV3"]:
    history_files = [h for h in dir_files if "history" in h and model_name in h]
    layers = [h.split("_")[1] for h in history_files]

    final_accuracies = dict()

    for layer, file in zip(layers, history_files):
        with open(os.path.join(probe_experiment_dir,model_name + '_' + layer + '_history.pkl'), 'rb') as f:
            training_history = pickle.load(f)

        final_accuracies[layer] = [training_history['sparse_categorical_accuracy'][-1],
                                   training_history['val_sparse_categorical_accuracy'][-1]]

        fig = plt.figure(figsize=(1.7,1))
        plt.plot(training_history['sparse_categorical_accuracy'])
        plt.plot(training_history['val_sparse_categorical_accuracy'])
    #     plt.title('layer' + str(i).zfill(3))
        plt.ylim(0,1)
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["training","validation"])
        plt.savefig(os.path.join(probe_experiment_dir, "plots", 'accuracy_' + model_name + '_' + layer + ".pdf"), bbox_inches='tight')
        plt.close(fig)

    sorted_layers = np.sort([*final_accuracies.keys()])
    accuracies_stacked = list()
    for layer in sorted_layers:
        accuracies_stacked.append(final_accuracies[layer])
    accuracies_stacked = np.stack(accuracies_stacked)

    fig, ax = plt.subplots()
    fig.set_size_inches(4,1.7)
    plt.bar(np.arange(accuracies_stacked.shape[0])-0.2,
            accuracies_stacked[:,0],
            width=0.4)
    plt.bar(np.arange(accuracies_stacked.shape[0])+0.2,
            accuracies_stacked[:, 1],
            width=0.4)
    ax.legend(["training","validation"], loc='upper center',
              ncol=2,
              bbox_to_anchor=(0.5, 1.25)
              )
    plt.ylim(0,1)
    if model_name == "InceptionV3":
        plt.xticks(np.arange(0,15,2))
    elif model_name == "VGG16":
        plt.xticks(np.arange(0, 8, 1))
    elif model_name == "ResNet50":
        plt.xticks([0]+list(np.arange(1, 20, 2)))
    plt.ylabel("final accuracy")
    plt.xlabel("layer ID")
    plt.grid(axis='y')
    plt.savefig(os.path.join(probe_experiment_dir, "plots", 'accuracy_' + model_name + "_overview.pdf"), bbox_inches='tight')

