from tqdm import tqdm

from experiments.jair.specific_function import plot_frequencies
from src.utils import makedirs
import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json

def image_path_to_numpy(file_name):
    img = Image.open(file_name)
    img.load()
    img_array = np.asarray(img, dtype="float32")
    return img_array

config_file = "experiments/jair/preprocess_fairface.json"
with open(config_file, "r") as f:
    config = json.load(f)

data_base_dir = config["raw_data_path"]
data_out_dir = config["processed_data_path"]

makedirs([data_out_dir])

protected_variables = ["race", "age", "gender"]
batch_size = 128
n_random_per_class = 5*batch_size

data_split_subset = "train"
raw_data_path = os.path.join(data_base_dir, data_split_subset)

onlyfiles = [f for f in os.listdir(raw_data_path) if os.path.isfile(os.path.join(raw_data_path, f))]

# training labels are given in cvs file
label_csv_path = os.path.join(data_base_dir, 'fairface_label_' + data_split_subset + '.csv')
label_df = pd.read_csv(label_csv_path)
label_df = label_df.replace("more than 70", value=">70")
label_df = label_df.replace("0-2", value="00-02")
label_df = label_df.replace("3-9", value="03-09")
lb_make = LabelEncoder()
# transform string to int

label_df["intersection"] = (label_df[protected_variables[0]].astype(str) + "#"
                            + label_df[protected_variables[1]].astype(str) + "#"
                            + label_df[protected_variables[2]].astype(str))
# train_labels = dict()
# for protected_variable in protected_variables:
#     label_df[protected_variable+"_int"] = lb_make.fit_transform(label_df[protected_variable])
#     train_labels[protected_variable] = label_df[protected_variable+"_int"].values
label_df["intersection_int"] = lb_make.fit_transform(label_df["intersection"])
train_labels = label_df["intersection_int"].values

class_name_to_idx = dict()
for protected_variable in protected_variables:
    label_as_index = lb_make.fit_transform(label_df[protected_variable])
    identifier_string = (label_df[protected_variable].astype(str) + ";"
                         + label_as_index.astype(str))
    class_idx_and_names = np.unique(identifier_string)
    class_idx_and_names_array = np.array([s.split(';') for s in class_idx_and_names])
    class_name_to_idx[protected_variable] = dict(zip(class_idx_and_names_array[:,0],
                                                     class_idx_and_names_array[:,1].astype('int32').tolist()
                                                     ))


identifier_string = (label_df["intersection_int"].astype(str) + ";"
                     + label_df["intersection"].astype(str))
class_idx_and_names = np.unique(identifier_string)
class_idx_to_name = np.array([s.split(';') for s in class_idx_and_names])
# class_idx_to_name = dict(zip(class_idx_to_name[:,0],class_idx_to_name[:,1]))

variable_frequencies = np.zeros((len(class_name_to_idx["race"].values()),
                                 len(class_name_to_idx["age"].values()),
                                 len(class_name_to_idx["gender"].values())))

corpus_file_content = []
batch_id = 0
for c_id, c_name in tqdm(class_idx_to_name):
    race, age, gender = c_name.split("#")

    class_example_ids = np.argwhere(train_labels == int(c_id))[:,0]
    variable_frequencies[class_name_to_idx["race"][race], class_name_to_idx["age"][age], class_name_to_idx["gender"][gender]] = len(class_example_ids)


    if len(class_example_ids) < n_random_per_class:
        pick_n = len(class_example_ids)
    else:
        pick_n = n_random_per_class
    random_class_ids = class_example_ids[np.random.choice(len(class_example_ids),
                                                          pick_n, replace=False)]

    batch_start_idx = 0

    while batch_start_idx < n_random_per_class and len(class_example_ids) > batch_start_idx:
        # load (up to) batch_size examples

        class_example_ids_batch = random_class_ids[batch_start_idx:batch_start_idx+batch_size]
        image_batch = list()
        for class_example_id in class_example_ids_batch:
            image_array = image_path_to_numpy(os.path.join(data_base_dir,
                                                           label_df["file"][class_example_id]))
            image_batch.append(image_array)
        image_batch = np.stack(image_batch)

        batch_name = 'batch' + str(batch_id).zfill(4)
        np.save(os.path.join(data_out_dir, batch_name), image_batch)
        corpus_file_content.append([batch_name + '.npy', race, age, gender, str(c_id)])

        batch_start_idx = batch_start_idx + batch_size
        batch_id = batch_id + 1

with open('experiments/jair/class_name_to_idx.json', 'w', encoding='utf-8') as f:
    json.dump(class_name_to_idx, f, ensure_ascii=False, indent=4)
np.save("experiments/jair/variable_frequencies.npy", variable_frequencies)

plot_frequencies("experiments/jair/")
plot_frequencies("experiments/jair/", n_random_per_class)

corpus_file_content = np.array(corpus_file_content)
np.savetxt(os.path.join(data_out_dir, 'corpus.csv'), corpus_file_content, delimiter=",", fmt='%s')

