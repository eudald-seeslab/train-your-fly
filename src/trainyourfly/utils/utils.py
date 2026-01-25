import json
import os
import random
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import torch

from trainyourfly.utils.debug_utils import model_summary


def get_files_from_directory(directory_path):
    files = []
    for root, _, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith((".npy", ".png", ".jpg", ".jpeg")):
                files.append(os.path.join(root, filename))
    return files


def get_image_paths(images_dir, small_length=None):
    images = get_files_from_directory(images_dir)
    assert len(images) > 0, f"No images found in {images_dir}."

    if small_length is not None:
        try:
            images = random.sample(images, small_length)
        except ValueError:
            print(
                f"Not enough images in {images_dir} to sample {small_length}."
                f"Continuing with {len(images)}."
            )

    return images


def synapses_to_matrix_and_dict(synapses):
    # Unique root_ids in synapse_df (both pre and post)
    neurons_synapse_pre = pd.unique(synapses["pre_root_id"])
    neurons_synapse_post = pd.unique(synapses["post_root_id"])
    all_neurons = np.unique(np.concatenate([neurons_synapse_pre, neurons_synapse_post]))

    # Map neuron root_ids to matrix indices
    root_id_to_index = {root_id: index for index, root_id in enumerate(all_neurons)}

    # Convert root_ids in filtered_synapse_df to matrix indices
    pre_indices = synapses["pre_root_id"].map(root_id_to_index).values
    post_indices = synapses["post_root_id"].map(root_id_to_index).values

    # Use syn_count as the data for the non-zero elements of the matrix
    data = synapses["syn_count"].values

    # Create the sparse matrix
    matrix = coo_matrix(
        (data, (pre_indices, post_indices)),
        shape=(len(all_neurons), len(all_neurons)),
        dtype=np.int64,
    )

    return matrix, root_id_to_index


def get_iteration_number(im_num, config_):
    if config_.debugging:
        return config_.debug_length
    small_length = config_.small_length
    if small_length is not None and im_num > small_length:
        return small_length // config_.batch_size
    return im_num // config_.batch_size


def get_label(name, classes):
    x = os.path.basename(os.path.dirname(name))
    try:
        return classes.index(x)
    except ValueError:
        raise ValueError(f"Unexpected directory label '{x}'")


def paths_to_labels(paths, classes):
    return [get_label(a, classes) for a in paths]


def select_random_images(all_files, batch_size, already_selected):
    # Filter out files that have already been selected
    remaining_files = [f for f in all_files if f not in already_selected]

    # Select batch_size random videos from the remaining files
    selected_files = random.sample(
        remaining_files, min(batch_size, len(remaining_files))
    )

    # Update the already_selected list
    already_selected.extend(selected_files)

    return selected_files, already_selected


def compute_accuracy(probabilities, labels):

    # Convert probabilities to binary predictions
    predictions = (probabilities > 0.5).float()

    # Calculate accuracy
    return np.where(predictions == labels, 1, 0).float().mean()


def update_running_loss(loss_, inputs_):
    return loss_.item() * inputs_.size(0)


def update_results_df(
    results_, batch_files_, outputs_, predictions_, batch_labels_, correct_
):
    return pd.concat(
        [
            results_,
            pd.DataFrame(
                {
                    "Image": batch_files_,
                    "Model outputs": list(outputs_),
                    "Prediction": predictions_,
                    "True label": batch_labels_,
                    "Is correct": correct_,
                }
            ),
        ]
    )


def initialize_results_df():
    return pd.DataFrame(
        columns=["Image", "Model outputs", "Prediction", "True label", "Is correct"]
    )


def clean_model_outputs(outputs_, batch_labels_):
    probabilities_ = torch.softmax(outputs_.detach().cpu().float(), dim=1).numpy()
    predictions_ = np.argmax(probabilities_, axis=1)
    batch_labels_cpu = batch_labels_.detach().cpu().numpy()
    correct_ = np.where(predictions_ == batch_labels_cpu, 1, 0)

    return probabilities_, predictions_, batch_labels_cpu, correct_


def module_to_clean_dict(module_):
    return {
        key: str(value)
        for key, value in module_.__dict__.items()
        if not key.startswith("__") and not "module" in str(value)
    }


def save_checkpoint(model_, optimizer_, model_name, config_, epoch=None):
    if config_.debugging:
        return

    path_ = os.path.join(os.getcwd(), "models")

    if epoch is not None:
        model_name = model_name.replace(".pth", f"_{epoch}.pth")
        path_ = os.path.join(path_, "epoch_checkpoints")

    # create 'models' directory if it doesn't exist
    os.makedirs(path_, exist_ok=True)
    torch.save(
        {"model": model_.state_dict(), "optimizer": optimizer_.state_dict()},
        os.path.join(path_, model_name),
    )

    if epoch is not None and epoch > 0:
        return
    # create an accompanying config file
    # get the config module and create a dictionary from it
    config_dict = module_to_clean_dict(config_)

    # Get model summary information and add it to the config
    model_summary_info = model_summary(model_, print_=False)
    config_dict["model_summary"] = model_summary_info

    config_path = os.path.join(path_, model_name.replace(".pth", "_config.txt"))
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)


def update_config_with_sweep(config, sweep_config):
    if sweep_config is not None:
        for key, value in sweep_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Config does not have attribute {key}")
    return config


def process_warnings(u_config, logger):
    if u_config.debugging:
        logger.warning("Debugging mode is active.")
    if u_config.small_length is not None:
        logger.warning(f"Small_length is set to {u_config.small_length}.")
    if u_config.filtered_celltypes:
        logger.warning(
            f"Filtering neurons by the following cell types: {', '.join(u_config.filtered_celltypes)}."
        )
    if u_config.filtered_fraction is not None:
        logger.warning(
            f"Filtering a fraction of neurons: {u_config.filtered_fraction}."
        )
    if u_config.resume_checkpoint is not None:
        if u_config.num_epochs == 0:
            logger.warning(
                f"Resume_checkpoint is not None and num_epochs is 0. "
                f"The model will not be trained, only evaluated."
            )
        else:
            logger.warning(f"Resuming training from {u_config.resume_checkpoint}")
    else:
        if u_config.num_epochs == 0:
            raise ValueError(
                "num_epochs is 0 and resume_checkpoint is None. "
                "Please set num_epochs to a positive integer."
            )
    if u_config.num_decision_making_neurons is not None:
        logger.warning(
            f"Using only {u_config.num_decision_making_neurons} neurons for decision making."
        )
