import os.path
import shutil

pipeline_dir_input_dependencies = {
    "align": ["acts","grads"],
    "naps": ["acts"],
    "naps_aligned": ["aligned"],
    "instance_projection": ["acts"],
    "instance_projection_plots": ["instance_projection_data"],
    "contrastive_naps": ["naps"],
    "topomap_layout": ["naps"],
    "topomap_layout_contrastive": ["contrastive_naps"],
    "topomap_activations": ["naps"],
    "topomap_activations_contrastive": ["contrastive_naps"],
    "topomap_plots": ["topomap_data"]
}


assure_sync = True

pipeline_dir_async_effects = {
    "align": ["naps", "contrastive_naps", "topomap_data", "topomap_plots"],
    "naps": ["contrastive_naps", "topomap_data", "topomap_plots"],
    "naps_aligned": ["contrastive_naps", "topomap_data", "topomap_plots"],
    "instance_projection": ["instance_projection_plots"],
    "instance_projection_plots": [],
    "contrastive_naps": ["topomap_data", "topomap_plots"],
    "topomap_layout": ["topomap_plots"],
    "topomap_layout_contrastive": ["topomap_plots"],
    "topomap_activations": ["topomap_plots"],
    "topomap_activations_contrastive": ["topomap_plots"],
    "topomap_plots": []
}

def get_user_delete_agreement():
    print('Do you agree with deleting potentially dependent directories? (yes/no):')
    x = input()
    if x != "yes":
        stop_msg = "Stopping: either agree to deleting or set 'assure_sync=False' in src/checks.py ."
        raise ValueError(stop_msg)
    else:
        return True

def check_pipeline_dependencies(processed_corpus_path, step):

    for dir_name in pipeline_dir_input_dependencies[step]:
        check_path = os.path.join(processed_corpus_path, dir_name)
        if not os.path.isdir(check_path):
            err_msg = "step '" + step + "' depends on '" + dir_name + "' which does not exist. "
            raise ValueError(err_msg)

    if assure_sync:
        agreed_deleting = False
        for dir_name in pipeline_dir_async_effects[step]:
            check_path = os.path.join(processed_corpus_path, dir_name)
            if os.path.isdir(check_path):
                info_msg = ("running step '" + step + "' overwrites data that was used for creating '" + dir_name + "'.\n" +
                            "Needs to be deleted according to 'assure_sync=True' flag in src/checks.py .")
                print(info_msg)
                if not agreed_deleting:
                    agreed_deleting = get_user_delete_agreement()
                shutil.rmtree(check_path)