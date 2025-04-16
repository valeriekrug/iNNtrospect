import os
import sys

sys.path.insert(0, ".")
import numpy as np
import pickle
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

_, model_name, layer = sys.argv
print(model_name, layer)

probe_base_path = "probes/intersectional/"

classifier_probes_data_file = probe_base_path + 'classifier_probes_data.pkl'

activations_dir = "output/fairface/" + model_name + "/acts/" + layer

if not os.path.isfile(classifier_probes_data_file):
    corpus_dir = "processed_data/fairface/"
    corpus_file = os.path.join(corpus_dir, "corpus.csv")
    batches = []
    batch_labels = []
    batch_label_names = []
    with open(corpus_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            batch, race, age, gender, label = line.split(',')
            batches.append(batch)
            batch_labels.append(int(label))
            batch_label_names.append("#".join([race, age, gender]))
    n_labels = len(np.unique(batch_labels))

    # pick one batch of each label
    # take 90 percent as training and 10 percent as valid
    data_info = dict()
    for label in range(n_labels):
        data_info[label] = dict()

        idx_of_label = np.argwhere(np.array(batch_labels) == label)[:, 0]
        # encourage picking full batches
        if len(idx_of_label) > 1:
            idx_of_label = idx_of_label[:-1]
        random_batch_with_label = np.random.choice(idx_of_label, 1)[0]

        batch_file_name = batches[random_batch_with_label]
        data_info[label]['file'] = batch_file_name
        data_info[label]['class_name'] = batch_label_names[random_batch_with_label]

        batch_size = np.load(os.path.join(corpus_dir, batch_file_name)).shape[0]
        id_permutation = np.random.permutation(np.arange(batch_size))
        split_at = np.floor(batch_size * 0.9).astype('int')
        data_info[label]['train_ids'] = id_permutation[:split_at]
        data_info[label]['valid_ids'] = id_permutation[split_at:]

    with open(classifier_probes_data_file, 'wb') as f:
        pickle.dump(data_info, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(classifier_probes_data_file, 'rb') as handle:
        data_info = pickle.load(handle)

# make data_info to training data
print("building data")

subsample_mode = "pooling"
if subsample_mode not in ["pooling", "grid"]:
    raise ValueError('subsample_mode must be from ["pooling","grid"], but is ' + subsample_mode )
first_batch_acts = np.load(os.path.join(activations_dir, data_info[0]['file']))
if len(first_batch_acts.shape) > 2:
    resolution_factor = int(np.ceil(first_batch_acts.shape[1] / 8))
    if subsample_mode == "pooling":
        subsample_model = keras.Sequential([
            keras.layers.AveragePooling2D(input_shape=first_batch_acts.shape[1:],
                                          pool_size=(resolution_factor, resolution_factor)),
        ])
        subsample_model.summary()


training_data = list()
training_labels = list()
valid_data = list()
valid_labels = list()
for label in tqdm(data_info.keys()):
    label_data_info = data_info[label]
    batch_acts = np.load(os.path.join(activations_dir, label_data_info['file']))

    # subsample image/feature map dimension to avoid huge amount of probe parameters
    if len(batch_acts.shape) > 2:
        if subsample_mode == "pooling":
            batch_acts = subsample_model(batch_acts).numpy()
        elif subsample_mode == "grid":
            resolution_factor = int(np.ceil(batch_acts.shape[1] / 8))
            start_from = resolution_factor // 2
            batch_acts = batch_acts[:, start_from::resolution_factor, start_from::resolution_factor, :]

    training_data.append(batch_acts[label_data_info['train_ids']])
    training_labels = training_labels + len(label_data_info['train_ids']) * [label]
    valid_data.append(batch_acts[label_data_info['valid_ids']])
    valid_labels = valid_labels + len(label_data_info['valid_ids']) * [label]

training_data = np.concatenate(training_data)
valid_data = np.concatenate(valid_data)

training_permutation = np.random.permutation(np.arange(len(training_labels)))
training_data = training_data[training_permutation]
training_labels = np.array(training_labels)[training_permutation]

valid_permutation = np.random.permutation(np.arange(len(valid_labels)))
valid_data = valid_data[valid_permutation]
valid_labels = np.array(valid_labels)[valid_permutation]

print("build and train model")

model = keras.Sequential([
    keras.layers.Flatten(input_shape=training_data.shape[1:]),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(len([*data_info.keys()]),
                       kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4))
])

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam()
metric = keras.metrics.SparseCategoricalAccuracy()

model.compile(optimizer, loss_fn, metrics=[metric])

model.summary()

outputs = model.fit(x=training_data,
                    y=training_labels,
                    validation_data=(valid_data,
                                     valid_labels),
                    epochs=50)

training_preds = np.argmax(model(training_data), 1)
training_confusion = confusion_matrix(training_labels, training_preds,
                                      labels=np.arange(126))
valid_preds = np.argmax(model(valid_data), 1)
valid_confusion = confusion_matrix(valid_labels, valid_preds,
                                   labels=np.arange(126))
confusion = np.stack([training_confusion, valid_confusion])
np.save(probe_base_path + model_name + '_' + layer + '_confusion.npy',
        confusion)

training_history = outputs.history
with open(probe_base_path + model_name + '_' + layer + '_history.pkl',
          'wb') as f:
    pickle.dump(training_history, f, protocol=pickle.HIGHEST_PROTOCOL)
