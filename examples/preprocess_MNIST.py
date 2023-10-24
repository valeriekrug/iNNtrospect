import os
from env_setup import ignore_user_installs, set_active_GPU

import numpy as np
import tensorflow as tf

ignore_user_installs('akrug')
set_active_GPU(0)

output_path = '/data/project/iNNtrospect/MNIST/'
batch_size = 128
n_random_per_class = 200

if not os.path.isdir(output_path):
    os.makedirs(output_path)

(_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data('/data/project/MNIST/mnist.npz')

test_images_rescaled = (test_images.astype('float32') / 255.)[..., np.newaxis]

# exam_indices = np.random.choice(10000,100)
# exam_tasks = test_images[exam_indices]

# class_to_example_ids = list()
corpus_file_content = []
batch_id = 0
for cid in range(10):
    class_example_ids = np.argwhere(test_labels==cid)[:,0]
    # class_to_example_ids.append(class_example_ids)

    random_class_ids = class_example_ids[np.random.choice(len(class_example_ids),n_random_per_class)]

    batch_start_idx = 0

    while batch_start_idx < n_random_per_class:
        image_batch = test_images_rescaled[random_class_ids[batch_start_idx:batch_start_idx+batch_size]]

        batch_name = 'batch' + str(batch_id).zfill(4)
        np.save(os.path.join(output_path, batch_name), image_batch)
        corpus_file_content.append([batch_name+'.npy', str(cid), cid])

        batch_start_idx = batch_start_idx + batch_size
        batch_id = batch_id + 1

corpus_file_content = np.array(corpus_file_content)
np.savetxt(os.path.join(output_path, 'corpus.csv'), corpus_file_content, delimiter=",", fmt='%s')





