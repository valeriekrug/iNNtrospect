import json
import os

from matplotlib import patches, pyplot as plt, lines
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from scipy.spatial import distance_matrix
from scipy.special import rel_entr
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from constants.check_constants import PIPELINE_STEPS
from constants.directory_constants import OUTPUT_DIRECTORY_NAMES
from src.checks import check_pipeline_dependencies
from src.data_processing import create_acts_grads_output_dirs, get_acts_and_grads_of_batch, \
    save_batch_acts_and_grads_per_layer
import numpy as np

from src.utils import makedirs
from src.visualization import get_absmax_clim


def process_intersectional_corpus_file(data_path, model, preprocessing_fun, output_path):
    check_pipeline_dependencies(output_path, PIPELINE_STEPS.ACTS_GRADS)

    create_acts_grads_output_dirs(output_path, len(model.outputs))

    corpus_path = os.path.join(data_path, "corpus.csv")
    corpus = np.loadtxt(corpus_path, dtype=str, delimiter=',')

    group_to_file_dict = dict()

    # individual variables
    for i in range(3):
        label_names = corpus[:, i+1]
        group_names = np.unique(label_names)
        for group_name in group_names:
            ids_of_group_files = np.argwhere(label_names == group_name)[:,0]
            group_to_file_dict[group_name] = list(corpus[ids_of_group_files,0])

    # paired variables
    for i,j in [[1,2],[1,3],[2,3]]:
        var1 = corpus[:,i]
        var2 = corpus[:,j]
        label_names = np.array(["#".join([c1,c2]) for c1,c2 in zip(var1,var2)])
        group_names = np.unique(label_names)
        for group_name in group_names:
            ids_of_group_files = np.argwhere(label_names == group_name)[:, 0]
            group_to_file_dict[group_name] = list(corpus[ids_of_group_files, 0])


    # joint variables
    joined_label_names = np.array(["#".join(c) for c in corpus[:, 1:-1]])
    group_names = np.unique(joined_label_names)

    for group_name in group_names:
        ids_of_group_files = np.argwhere(joined_label_names == group_name)[:,0]
        group_to_file_dict[group_name] = list(corpus[ids_of_group_files,0])

    with open(os.path.join(output_path,
                           'group_name_to_files.json'), 'w') as f:
        json.dump(group_to_file_dict, f)

    all_group_names = [*group_to_file_dict.keys()]
    group_ids = np.arange(len(all_group_names)).tolist()
    index_to_group_name = dict(zip(group_ids, all_group_names))
    with open(os.path.join(output_path, "index_to_group_name.json"), "w") as f:
        json.dump(index_to_group_name, f)
    group_name_to_index = dict(zip(all_group_names, group_ids))
    with open(os.path.join(output_path, "group_name_to_index.json"), "w") as f:
        json.dump(group_name_to_index, f)

    for batch_info in tqdm(corpus, desc="save batch acts/grads"):
        batch_file = batch_info[0]

        batch = np.load(os.path.join(data_path, batch_file))
        batch_activations, batch_grads = get_acts_and_grads_of_batch(model,batch, preprocessing_fun)
        save_batch_acts_and_grads_per_layer(batch_activations, batch_grads, batch_file, output_path)


def draw_rect_with_text(ax, x, y, cmap, norm, value, width=10, height=10):
    rect = patches.Rectangle((x, y),
                             width, height,
                             linewidth=1, edgecolor='None',
                             facecolor=cmap(norm(value))
                             )
    ax.add_patch(rect)
    ax.text(x + width / 2,
            y + height / 2,
            f'{int(value):,}',
            fontsize=28,
            horizontalalignment='center',
            verticalalignment='center')


def draw_label_text(ax, x, y, text, side, width=10, height=10, fontsize=28):
    if side in ['top', 'bottom']:
        h_align = 'center'
        v_align = 'center'
    elif side == 'left':
        h_align = 'right'
        v_align = 'center'
    elif side == 'right':
        h_align = 'left'
        v_align = 'center'
    else:
        raise ValueError("side parameter must be top, bottom, left or right")

    ax.text(x + width / 2,
            y + height / 2,
            text,
            fontsize=fontsize,
            fontweight='bold',
            horizontalalignment=h_align,
            verticalalignment=v_align)

def plot_frequencies(data_path, max_count=None):
    freqs = np.load(os.path.join(data_path,"variable_frequencies.npy"))
    if max_count is not None:
        freqs[freqs > max_count] = max_count

    with open(os.path.join(data_path,"class_name_to_idx.json"), 'r', encoding='utf-8') as f:
        class_name_to_idx = json.load(f)
    class_names_per_dim = [[*class_name_to_idx["race"].keys()],
                           [*class_name_to_idx["age"].keys()],
                           [*class_name_to_idx["gender"].keys()]]

    plt.rcParams.update({'font.size': 32})

    fig, ax = plt.subplots()
    width_to_height = 7
    scale = 11
    fig.set_size_inches(1 * width_to_height * scale, 1 * scale)

    # Display an empty canvas
    ax.plot()
    for side in ['top', 'bottom', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    n_row, n_col, aspect_ratio = 8, 21, 0.5
    ax.set_box_aspect(aspect_ratio * n_row / n_col)

    cmap = ListedColormap(plt.get_cmap('YlOrRd_r')(np.linspace(0.05, 1, 256)))  # skip too dark colors
    if max_count is not None:
        max_freq = max_count
    else:
        max_freq = freqs.max()
        ceil_precision = 1000
        max_freq = np.ceil(max_freq / ceil_precision) * ceil_precision
    norm = plt.Normalize(0, max_freq)

    rect_width = 10
    rect_height = 10
    h_gaps = 40

    n_races, n_ages, n_genders = freqs.shape

    # draw the main contingency table blocks (one per gender)
    block_sums = np.sum(np.sum(freqs, 0), 0)
    for gender_id in range(n_genders):
        # draw group label of blocks
        x_label = n_ages * rect_width / 2 + gender_id * n_ages * rect_width + gender_id * h_gaps
        y_label = 1.8 * rect_height
        draw_label_text(ax, x_label, y_label, class_names_per_dim[2][gender_id], 'top')

        # draw invididual group counts
        for race_id in range(n_races):
            for age_id in range(n_ages):
                count = freqs[race_id, age_id, gender_id]
                x = age_id * rect_width + gender_id * n_ages * rect_width + gender_id * h_gaps
                y = -race_id * rect_height
                draw_rect_with_text(ax, x, y, cmap, norm, count)

        # compute and draw row sums and write group labels
        row_sums = np.sum(freqs[:, :, gender_id], 1)
        x_sum = n_ages * rect_width + gender_id * n_ages * rect_width + gender_id * h_gaps
        x_label = -0.6 * rect_width + gender_id * n_ages * rect_width + gender_id * h_gaps
        for race_id in range(n_races):
            y = -race_id * rect_height

            class_name = class_names_per_dim[0][race_id]
            draw_label_text(ax, x_label, y, class_name, 'left')

            row_sum = row_sums[race_id]
            draw_rect_with_text(ax, x_sum, y, cmap, norm, row_sum)

        # compute and draw column sums and write group labels
        col_sums = np.sum(freqs[:, :, gender_id], 0)
        y_sum = -n_races * rect_height
        y_label = 0.8 * rect_height

        for age_id in range(n_ages):
            x = age_id * rect_width + gender_id * n_ages * rect_width + gender_id * h_gaps

            class_name = class_names_per_dim[1][age_id]
            class_name = class_name.replace('00-02', '0-2')
            class_name = class_name.replace('03-09', '3-9')
            draw_label_text(ax, x, y_label, class_name, 'top')

            col_sum = col_sums[age_id]
            draw_rect_with_text(ax, x, y_sum, cmap, norm, col_sum)

        # compute and draw block sums
        block_sum = block_sums[gender_id]
        x = n_ages * rect_width + gender_id * n_ages * rect_width + gender_id * h_gaps
        y = -n_races * rect_height
        draw_rect_with_text(ax, x, y, cmap, norm, block_sum)

        # draw lines to separate individual counts from row/column sums
        # reusing x_sum and y_sum from the column sums and row sums, respectively
        line = lines.Line2D([x_sum, x_sum],
                            [rect_height, -n_races * rect_height],
                            linewidth=5,
                            color='black')
        ax.add_artist(line)

        line = lines.Line2D([gender_id * n_ages * rect_width + gender_id * h_gaps,
                             gender_id * n_ages * rect_width + gender_id * h_gaps + (n_ages + 1) * rect_width],
                            [y_sum + rect_height, y_sum + rect_height],
                            linewidth=5,
                            color='black')
        ax.add_artist(line)

    # draw row- and column-sums of both blocks and write group labels
    # total row sums
    total_row_sums = np.sum(np.sum(freqs, 1), 1)
    x_sum = n_genders * n_ages * rect_width + n_genders * h_gaps
    x_label = -0.6 * rect_width + n_genders * n_ages * rect_width + n_genders * h_gaps
    for race_id in range(n_races):
        y = -race_id * rect_height

        class_name = class_names_per_dim[0][race_id]
        draw_label_text(ax, x_label, y, class_name, 'left')

        row_sum = total_row_sums[race_id]
        draw_rect_with_text(ax, x_sum, y, cmap, norm, row_sum)

    # total column sums
    total_col_sums = np.sum(np.sum(freqs, 0), 1)
    y_sum = -(n_races + 2) * rect_height
    y_label = -(n_races + 1.2) * rect_height
    for age_id in range(n_ages):
        x = age_id * rect_width + 5 * rect_width

        class_name = class_names_per_dim[1][age_id]
        class_name = class_name.replace('00-02', '0-2')
        class_name = class_name.replace('03-09', '3-9')
        draw_label_text(ax, x, y_label, class_name, 'top')

        col_sum = total_col_sums[age_id]
        draw_rect_with_text(ax, x, y_sum, cmap, norm, col_sum)

    # total sum
    total_sum = np.sum(freqs)
    x_sum = n_genders * n_ages * rect_width + n_genders * h_gaps
    x_label = -0.6 * rect_width + n_genders * n_ages * rect_width + n_genders * h_gaps
    y = -(n_races + 2) * rect_height
    draw_label_text(ax, x_label, y, "total count", 'left')
    draw_rect_with_text(ax, x_sum, y, cmap, norm, total_sum)

    ax.set_xlim([-0.6 * h_gaps,
                 n_genders * n_ages * rect_width + n_genders * h_gaps + rect_width])
    ax.set_ylim([-(n_races + 2) * rect_height,
                 2.8 * rect_height])

    fig.colorbar(ScalarMappable(cmap=cmap, norm=norm),
                 ax=ax,
                 label="count",
                 pad=0.01,
                 shrink=0.9)

    plot_file_name = "variable_frequencies"
    if max_count is not None:
        plot_file_name = plot_file_name + "_" + str(int(max_count))
    plot_file_name = plot_file_name + ".pdf"

    plt.savefig(os.path.join(data_path, plot_file_name), bbox_inches='tight')
    plt.close(fig)

def plot_topomaps_intersectionally(experiments_path, processed_corpus_path):
    check_pipeline_dependencies(processed_corpus_path, PIPELINE_STEPS.TOPOMAP_PLOTS)
    topomap_data_dir = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.TOPOMAP_DATA)

    plot_output_dir = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.TOPOMAP_PLOTS)
    makedirs([plot_output_dir])

    with open(os.path.join(experiments_path,'class_name_to_idx.json'), 'r', encoding='utf-8') as f:
        class_name_to_idx = json.load(f)
    class_names_per_dim = [[*class_name_to_idx["race"].keys()],
                           [*class_name_to_idx["age"].keys()],
                           [*class_name_to_idx["gender"].keys()]]

    resolution = 100
    h_gap_size = 170
    v_gap_size = 80
    topomap_padding = 10

    n_races, n_ages, n_genders = [len(n) for n in class_names_per_dim]

    image_width = (2 * n_ages + 3) * resolution + n_ages * 2 * topomap_padding + n_genders * h_gap_size
    image_height = (n_races + 2) * resolution + n_races * topomap_padding + v_gap_size

    topomap_bg = np.zeros([image_height, image_width])

    block_width = (n_ages + 1) * resolution + n_ages * topomap_padding
    cell_width_height = resolution + topomap_padding

    act_group_names = np.load(os.path.join(topomap_data_dir, "group_names.npy"))


    topomap_data_files = os.listdir(topomap_data_dir)
    interpolated_activations_files = [p for p in topomap_data_files if "interpolated" in p]

    for filename in interpolated_activations_files:
        layer_name = filename.split('_')[0]
        interpolated_acts = np.load(os.path.join(topomap_data_dir,filename))

        ## build a large matrix to visualize as background image
        for gender_id in range(n_genders):
            for race_id in range(n_races):
                for age_id in range(n_ages):
                    group_name = class_names_per_dim[0][race_id] + "#" + class_names_per_dim[1][age_id] + "#" + \
                                 class_names_per_dim[2][gender_id]
                    index_of_group = np.where(act_group_names == group_name)[0][0]
                    topomap_values = interpolated_acts[index_of_group]

                    x = age_id * cell_width_height + gender_id * (h_gap_size + block_width)  # x-coordinate, left to right
                    y = race_id * cell_width_height  # y-coordinate, top to bottom

                    topomap_bg[y:y + resolution, x:x + resolution] = topomap_values

        # "row sums": combination of gender with race
        for gender_id in range(n_genders):
            for race_id in range(n_races):
                group_name = class_names_per_dim[0][race_id] + "#" + class_names_per_dim[2][gender_id]
                index_of_group = np.where(act_group_names == group_name)[0][0]
                topomap_values = interpolated_acts[index_of_group]

                x = n_ages * cell_width_height + gender_id * (h_gap_size + block_width)  # x-coordinate, left to right
                y = race_id * cell_width_height  # y-coordinate, top to bottom

                topomap_bg[y:y + resolution, x:x + resolution] = topomap_values

        # # "total row sums": race
        x = n_genders * (h_gap_size + block_width)  # x-coordinate, left to right
        for race_id in range(n_races):
            group_name = class_names_per_dim[0][race_id]
            index_of_group = np.where(act_group_names == group_name)[0][0]
            topomap_values = interpolated_acts[index_of_group]

            y = race_id * cell_width_height  # y-coordinate, top to bottom

            topomap_bg[y:y + resolution, x:x + resolution] = topomap_values

        # "col sums": combination of gender with age
        y = n_races * cell_width_height  # y-coordinate, top to bottom
        for gender_id in range(n_genders):
            for age_id in range(n_ages):
                group_name = class_names_per_dim[1][age_id] + "#" + class_names_per_dim[2][gender_id]
                index_of_group = np.where(act_group_names == group_name)[0][0]
                topomap_values = interpolated_acts[index_of_group]

                x = age_id * cell_width_height + gender_id * (h_gap_size + block_width)  # x-coordinate, left to right

                topomap_bg[y:y + resolution, x:x + resolution] = topomap_values

        # "total col sums": age
        x_indent = (block_width + h_gap_size) // 2  # x-coordinate, left to right
        y = n_races * cell_width_height + resolution + v_gap_size  # y-coordinate, top to bottom
        for age_id in range(n_ages):
            group_name = class_names_per_dim[1][age_id]
            index_of_group = np.where(act_group_names == group_name)[0][0]
            topomap_values = interpolated_acts[index_of_group]

            x = age_id * cell_width_height + x_indent  # x-coordinate, left to right

            topomap_bg[y:y + resolution, x:x + resolution] = topomap_values

        # "block sums": gender
        y = n_races * cell_width_height  # y-coordinate, top to bottom
        for gender_id in range(n_genders):
            group_name = class_names_per_dim[2][gender_id]
            index_of_group = np.where(act_group_names == group_name)[0][0]
            topomap_values = interpolated_acts[index_of_group]

            x = n_ages * cell_width_height + gender_id * (h_gap_size + block_width)

            topomap_bg[y:y + resolution, x:x + resolution] = topomap_values

        fig, ax = plt.subplots()
        fig.set_size_inches(25, 10)

        percentiles = np.percentile(topomap_bg,[0.5,99.5])
        clim = get_absmax_clim(percentiles)
        ax.imshow(topomap_bg, cmap='bwr', clim=clim)

        ## annotate lines on top of image

        for gender_id in range(n_genders):
            x_start = gender_id * (block_width + h_gap_size)
            x_end = x_start + block_width
            y_start = n_races * cell_width_height - topomap_padding // 2
            y_end = y_start
            line = lines.Line2D([x_start, x_end],
                                [y_start, y_end],
                                linewidth=2,
                                color='black')
            ax.add_artist(line)

            x_start = gender_id * (block_width + h_gap_size) + n_ages * cell_width_height - topomap_padding // 2
            x_end = x_start
            y_start = 0
            y_end = y_start + (n_races + 1) * cell_width_height - topomap_padding
            line = lines.Line2D([x_start, x_end],
                                [y_start, y_end],
                                linewidth=2,
                                color='black')
            ax.add_artist(line)

        ## annotate texts on top of image
        fontsize = 12
        plt.rcParams.update({'font.size': fontsize})

        for gender_id in range(n_genders + 1):
            for race_id in range(n_races):
                class_name = class_names_per_dim[0][race_id]
                class_name = class_name.replace(" ", "\n")
                class_name = class_name.replace("_", "\n")

                x = gender_id * (h_gap_size + block_width) - (
                            resolution + topomap_padding) // 2  # x-coordinate, left to right
                y = race_id * cell_width_height  # y-coordinate, top to bottom

                draw_label_text(ax, x, y, class_name, "left", resolution, resolution, fontsize)

        for gender_id in range(n_genders):
            for age_id in range(n_ages):
                class_name = class_names_per_dim[1][age_id]
                class_name = class_name.replace('00-02', '0-2')
                class_name = class_name.replace('03-09', '3-9')
                x = age_id * cell_width_height + gender_id * (h_gap_size + block_width)
                y = -resolution // 1.5

                draw_label_text(ax, x, y, class_name, "top", resolution, resolution, fontsize)

        x_indent = (block_width + h_gap_size) // 2  # x-coordinate, left to right
        for age_id in range(n_ages):
            class_name = class_names_per_dim[1][age_id]
            class_name = class_name.replace('00-02', '0-2')
            class_name = class_name.replace('03-09', '3-9')

            x = age_id * cell_width_height + x_indent  # x-coordinate, left to right
            y = n_races * cell_width_height + resolution + v_gap_size - resolution // 1.5  # y-coordinate, top to bottom

            draw_label_text(ax, x, y, class_name, "top", resolution, resolution, fontsize)

        for gender_id in range(n_genders):
            class_name = class_names_per_dim[2][gender_id]

            x = gender_id * (h_gap_size + block_width) + (n_ages * cell_width_height - topomap_padding) // 2
            y = -resolution // 0.9

            draw_label_text(ax, x, y, class_name, "top", resolution, resolution, fontsize)

        for side in ['top', 'bottom', 'right', 'left']:
            ax.spines[side].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([-h_gap_size, n_genders * (h_gap_size + block_width) + resolution])
        ax.set_ylim([(n_races + 2) * cell_width_height + v_gap_size, -resolution // 1.1])

        fig.colorbar(ScalarMappable(cmap='bwr', norm=plt.Normalize(clim[0], clim[1])),
                     ax=ax,
                     label="NAP value",
                     orientation="vertical",
                     pad=0.01,
                     shrink=0.7,
                     anchor=(0, 0.7))

        fig.savefig(os.path.join(plot_output_dir, 'topomap_overview_'+layer_name+'.pdf'), bbox_inches='tight')
        plt.close(fig)


def compute_inter_group_distmat(processed_corpus_path, layer, group_names, group_name_to_files, output_dir):
    # inter-group activation distances

    inter_dist_mat_file_path = os.path.join(output_dir, layer + ".npy")
    if os.path.isfile(inter_dist_mat_file_path):
        inter_dist_mat = np.load(inter_dist_mat_file_path)
    else:

        example_batch = os.path.join(processed_corpus_path, 'acts', layer,
                     group_name_to_files[group_names[0]][0])
        example_activations = np.load(example_batch)
        resolution_factor = int(np.ceil(example_activations.shape[1] / 8))
        start_from = resolution_factor // 2
        print(layer,
              example_activations.shape,
              ", each", resolution_factor,
              ", start at" , start_from,
              ", end at", start_from + resolution_factor * (len(np.arange(start_from,example_activations.shape[1],resolution_factor))-1),
              ", len", len(np.arange(start_from,example_activations.shape[1],resolution_factor)))


        # inter_dist_distributions = dict()
        inter_dist_mat = np.zeros([len(group_names), len(group_names)])

        for start_id, groupA in enumerate(group_names):
            batch_files = group_name_to_files[groupA]

            first_batch_file = batch_files[0]
            first_batch_path = os.path.join(processed_corpus_path, 'acts', layer,
                                            first_batch_file)

            group_activations = np.load(first_batch_path)

            if len(group_activations.shape)>2:
                # decrease resolution
                group_activations = group_activations[:, start_from::resolution_factor, start_from::resolution_factor, :]


            flat_group_activationsA = np.reshape(group_activations,
                                                 [group_activations.shape[0],
                                                  np.prod(group_activations.shape[1:])])

            #     for end_id, groupB in tqdm(enumerate(group_names[(start_id+1):]), desc=groupA):
            for end_id, groupB in tqdm(enumerate(group_names[start_id:]), desc=str(start_id)):
                batch_files = group_name_to_files[groupB]

                first_batch_file = batch_files[0]
                first_batch_path = os.path.join(processed_corpus_path, 'acts', layer,
                                                first_batch_file)

                group_activations = np.load(first_batch_path)
                if len(group_activations.shape) > 2:
                    # decrease resolution
                    group_activations = group_activations[:, start_from::resolution_factor, start_from::resolution_factor, :]
                flat_group_activationsB = np.reshape(group_activations,
                                                     [group_activations.shape[0],
                                                      np.prod(group_activations.shape[1:])])

                dist = distance_matrix(flat_group_activationsA, flat_group_activationsB)
                dist_flat = np.reshape(dist, np.prod(dist.shape))
                # inter_dist_distributions[groupA + ";" + groupB] = dist_flat
                avg_dist = np.mean(dist_flat)
                inter_dist_mat[start_id, end_id + start_id] = avg_dist
                inter_dist_mat[end_id + start_id, start_id] = avg_dist

        np.save(os.path.join(output_dir, layer + ".npy"), inter_dist_mat)

    return inter_dist_mat

def plot_stacked_bars_reference(group_names, variable_names, class_names_per_dim, output_dir):
    xlim = [-0.25, 1.9]

    groups_in_cluster = np.array(group_names)
    variables_in_cluster = np.array([g.split('#') for g in groups_in_cluster])

    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(7.4, 5)

    for column_id, variable in enumerate(variable_names):
        g_names = class_names_per_dim[column_id]
        cluster_counts = get_counts_of_classes(variables_in_cluster[:, column_id],
                                               g_names)

        bottom = 0
        for group, count in zip(g_names, cluster_counts):
            print_group = group.replace('00-02', '0-2')
            print_group = print_group.replace('03-09', '3-9')
            print_group = print_group.replace(' ', '\n')
            print_group = print_group.replace('_', '\n')
            axes[column_id].bar(variable, count, 0.5, label=print_group, bottom=bottom)
            bottom = bottom + count

        axes[column_id].legend(loc="right", reverse=True)
        axes[column_id].set_xlim(xlim)
        axes[column_id].set_ylim([0, np.sum(cluster_counts)])
        axes[column_id].set_yticks([])

    plt.savefig(os.path.join(output_dir, 'reference.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_stacked_bars(clustering, group_names, variable_names, class_names_per_dim, output_dir, layer):

    fig, axes = plt.subplots(2, 5)
    axes = axes.reshape([10])
    fig.set_size_inches(8*1.5, 5*1.5)

    n_options = [7, 9, 2]
    # n_probs = []
    # for n in n_options:
    #     n_probs.append(n * [1 / n])
    kl_divs = np.zeros([2, 3, 10])

    max_kls = list()
    for n_option in n_options:
        expected_probs = np.array(n_option * [1])
        expected_probs = expected_probs / np.sum(expected_probs)
        max_kl = np.sum(rel_entr([1] + (len(expected_probs) - 1) * [0], expected_probs))
        max_kls.append(max_kl)
    max_kls = np.array(max_kls)

    counts = np.zeros([10, 11, 3])
    cluster_sizes = np.zeros(10,'int32')
    for cid in range(10):
        group_in_cluster_idx = np.where(clustering.labels_ == cid)[0]
        groups_in_cluster = np.array(group_names)[group_in_cluster_idx]
        cluster_sizes[cid] = len(groups_in_cluster)
        variables_in_cluster = np.array([g.split('#') for g in groups_in_cluster])
        is_small_cluster = cluster_sizes[cid]<=3

        # counts = np.zeros([11, 3])
        for column_id, variable in enumerate(variable_names):
            g_names = class_names_per_dim[column_id]
            cluster_counts = get_counts_of_classes(variables_in_cluster[:, column_id],
                                                   g_names)
            counts[cid, :len(cluster_counts), column_id] = cluster_counts

            abs_counts = cluster_counts[:n_options[column_id]]
            rel_counts = abs_counts / np.sum(abs_counts)
            #         print(np.round(rel_counts,2), np.round(n_probs[column_id],2))
            # if is_small_cluster:
            #     kl_divs[column_id, cid] = np.nan
            # else:
            # kl_divs[column_id, cid] = np.sum(rel_entr(rel_counts,
            #                                       n_probs[column_id]))


            uniform_probs = n_options[column_id]*[1/n_options[column_id]]

            raw_KL = np.sum(rel_entr(rel_counts,
                                     uniform_probs))

            if cluster_sizes[cid] < n_options[column_id]:
                uniform_in_cluster = cluster_sizes[cid]*[1/cluster_sizes[cid]] + (len(uniform_probs) - cluster_sizes[cid]) * [0]
                min_KL = np.sum(rel_entr(uniform_in_cluster,
                                         uniform_probs))
            else:
                min_KL = 0
            if min_KL == max_kls[column_id]:
                norm_KL = 0
            else:
                norm_KL = (raw_KL-min_KL)/(max_kls[column_id]-min_KL)
            kl_divs[0,column_id, cid] = norm_KL
            if is_small_cluster:
                kl_divs[1, column_id, cid] = np.nan
            else:
                kl_divs[1, column_id, cid] = norm_KL

            # normalize by max possible KL depending on cluster size and variable options


        # kl_divs[:, cid] = kl_divs[:, cid] / max_kls
        # kl_div_cluster = np.sum(kl_divs[:, cid])

    kl_div_clusters = np.sum(kl_divs[0],0)
    order_by_kl = np.argsort(kl_div_clusters)[::-1]
    for cid in range(10):
        if cluster_sizes[order_by_kl[cid]]<=3:
            alpha = 0.5
            textcolor = '#AAAAAA'
        else:
            alpha = 1
            textcolor = 'black'

        xtick_labels = list(variable_names)
        kl_div_per_var = kl_divs[0, :, order_by_kl[cid]]
        for tick_idx in range(len(xtick_labels)):
            xtick_labels[tick_idx] = xtick_labels[tick_idx] + '\n' + f'({kl_div_per_var[tick_idx]:.2f})'

        bottom = np.zeros(3)
        for count in counts[order_by_kl[cid]]:
            axes[cid].tick_params(color=textcolor, labelcolor=textcolor)
            for spine in axes[cid].spines.values():
                spine.set_edgecolor(textcolor)

            axes[cid].bar(xtick_labels, count, 0.7, bottom=bottom, alpha=alpha)
            bottom = bottom + count

        axes[cid].set_title("cluster " + str(cid) + "\n$n$=" + str(cluster_sizes[order_by_kl[cid]]) + f', KL={kl_div_clusters[order_by_kl[cid]]:.2f}',
                            color=textcolor)
        axes[cid].set_yticks([])

    plt.subplots_adjust(hspace=0.35)
    plt.savefig(os.path.join(output_dir, layer + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    np.save(os.path.join(output_dir, layer + "_kl_divs.npy"), kl_divs)

def get_counts_of_classes(X, classes):
    counts = np.zeros(len(classes))
    for cid, c in enumerate(classes):
        counts[cid] = len(np.where(X==c)[0])
    return counts

def representation_cluster_plots(experiments_path, processed_corpus_path):
    with open(os.path.join(experiments_path, 'class_name_to_idx.json'), 'r', encoding='utf-8') as f:
        class_name_to_idx = json.load(f)
    variable_names = ["race", "age", "gender"]
    class_names_per_dim = list()
    for variable_name in variable_names:
        class_names_per_dim.append([*class_name_to_idx[variable_name].keys()])

    with open(os.path.join(processed_corpus_path, 'group_name_to_files.json'), 'r', encoding='utf-8') as f:
        group_name_to_files = json.load(f)
    group_names = [*group_name_to_files.keys()]
    group_names = [g for g in group_names if len(g.split('#')) == 3]

    activation_layers = os.listdir(os.path.join(processed_corpus_path, 'acts'))
    activation_layers = [a for a in activation_layers if "layer" in a]

    output_path_data = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.REPRESENTATION_CLUSTER_DATA)
    output_path_plots = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.REPRESENTATION_CLUSTER_PLOTS)
    makedirs([output_path_data, output_path_plots])
    plot_stacked_bars_reference(group_names,variable_names,class_names_per_dim, output_path_plots)

    for layer in np.sort(activation_layers):
        inter_dist_mat = compute_inter_group_distmat(processed_corpus_path, layer, group_names, group_name_to_files, output_path_data)

        scaled_distmat = inter_dist_mat - np.min(inter_dist_mat)
        scaled_distmat = np.max(scaled_distmat) - scaled_distmat
        clustering = AgglomerativeClustering(metric='euclidean',
                                             n_clusters=10,
                                             linkage='complete').fit(scaled_distmat)

        plot_stacked_bars(clustering, group_names, variable_names, class_names_per_dim, output_path_plots, layer)


def representation_cluster_bias_plots(processed_corpus_path):
    cluster_plot_dir = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.REPRESENTATION_CLUSTER_PLOTS)
    kl_div_files = os.listdir(cluster_plot_dir)
    kl_div_files = np.sort([f for f in kl_div_files if "npy" in f])

    all_kl_divs = []
    layer_names = []
    for f in kl_div_files:
        lname = f.split("_")[0]
        lname = lname.replace('layer','')
        while lname[0]=='0' and len(lname)>1:
            lname = lname[1:]
        layer_names.append(lname)
        all_kl_divs.append(np.load(os.path.join(cluster_plot_dir, f))[1])

    all_kl_divs = np.stack(all_kl_divs)
    cluster_average_kl_divs = np.nanmean(all_kl_divs, 2)

    fig, ax = plt.subplots()
    fig.set_size_inches(4,2)
    plt.plot(cluster_average_kl_divs)
    ax.legend(["race", "age", "gender"], loc='upper center',
              ncol=3,
              bbox_to_anchor=(0.5, 1.25)
              )
    if len(layer_names)>15:
        rotation = 90
    else:
        rotation = 0
    plt.xticks(np.arange(len(layer_names)), layer_names, rotation=rotation)
    plt.ylim([0,np.max(cluster_average_kl_divs)*1.1])
    plt.ylabel("scaled KL-divergence")
    plt.xlabel("layer ID")
    plt.grid(linestyle='--')
    plt.savefig(os.path.join(cluster_plot_dir, 'KL_bias.pdf'), bbox_inches='tight')
    plt.close(fig)

