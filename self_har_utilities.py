import numpy as np
import sklearn
import gc
from torch.utils.data import Dataset, DataLoader
import torch
import argparse
import pickle
import data_pre_processing
import json
import os
import hashlib
# import mixer

__author__ = "Moh'd Khier Al Kfari"
__copyright__ = "Copyright (C) 2024 Moh'd Khier Al Kfari"

"""
Complementing the work of Al Kfari, Moh'D. Khier, and Lüdtke, Stefan: Domain Adaptation in Human Activity Recognition through Self-Training

This implementation is built upon and inspired by the work of Tang et al. in SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data.

@article{10.1145/3675094.3678465,
  author = {Al Kfari, Moh'D. Khier and Lüdtke, Stefan},
  title = {Domain Adaptation in Human Activity Recognition through Self-Training},
  year = {2024},
  issue_date = {2024},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  journal = {Companion of the 2024 on ACM International Joint Conference on Pervasive and Ubiquitous Computing},
  doi = {10.1145/3675094.3678465},
  abstract = {We investigate domain adaptation for Human Activity Recognition (HAR), where a model trained on one dataset (source) is applied to another dataset (target) with different characteristics. Specifically, we focus on evaluating the performance of SelfHAR, a recently introduced semi-supervised learning framework rooted in self-training. Unlike typical semi-supervised approaches that leverage unlabeled data to enhance model performance on a labeled dataset, our investigation centers on evaluating the performance gain on the unlabeled target data.

Our findings indicate that the SelfHAR algorithm can achieve performance levels nearly equivalent to supervised learning, achieving an F1 score of approximately 0.8 across datasets from different environments, even without labels for the target dataset. Furthermore, our approach consistently enhances performance compared to models trained solely on the source dataset, demonstrating its efficacy in adapting HAR models to diverse environmental conditions.},
  keywords = {domain adaptation, human activity recognition, self-training, semi-supervised learning, transfer learning}
}

Access to Article:
    https://doi.org/10.1145/3675094.3678465

Contact: mohd.kfari@uni-rostock.de

Copyright (C) 2024 M. K. Al Kfari

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
def data_loader(training_set, batch_size=1, shuffle=False):
    class HAR_Dataset(Dataset):
        def __init__(self, data):
            if len(data) ==3:
                x_data, y_data, datasets_flag = data  # Unpacking the tuple
            elif len(data) == 2:
                x_data, y_data = data  # Unpacking the tuple
                datasets_flag = torch.zeros(len(x_data))
            else:
                x_data = data
                y_data = torch.zeros(len(x_data))
                datasets_flag = torch.zeros(len(x_data))

            # Convert to Tensor
            self.x_data = torch.tensor(x_data, dtype=torch.float32)
            self.datasets_flag = torch.tensor(datasets_flag, dtype=torch.float32)
            if isinstance(y_data, dict):
                self.y_data = {key: torch.tensor(y_data[key], dtype=torch.float32) for key in y_data}
                # self.y_data = [torch.tensor(y_data[key], dtype=torch.float32) for key in y_data]
            else:
                self.y_data = torch.tensor(y_data, dtype=torch.float32)
        def __len__(self):
            return len(self.x_data)  # Number of samples

        def __getitem__(self, index):
            # Return x_data + a dictionary of task-specific labels
            if isinstance(self.y_data, dict): # Ensures correct format for multi-task models
                y_sample = {key: self.y_data[key][index] for key in self.y_data}
            else:
                y_sample = self.y_data[index]

            return self.x_data[index], y_sample, self.datasets_flag[index]

    return DataLoader(HAR_Dataset(training_set), batch_size=batch_size, shuffle=shuffle)

def initialize_and_load_model(model_weights_path, experiment_type, output_shape, input_shape, optimizer_type,
                              transform_funcs_names):
    """
    Initializes the model, loads pretrained weights, and prepares it for evaluation.

    Args:
        model_weights_path (str): Path to the trained model weights file.
        output_shape (int): The number of output classes.
        input_shape (tuple): The expected input shape of the model (sequence_length, num_channels).
        optimizer_type (str): The optimizer type to be used.

    Returns:
        torch.nn.Module: The model ready for evaluation.
    """

    # ✅ Select optimizer
    optimizer_instance = optimizer_selection(optimizer_type)

    # Initialize the base model
    core_model = self_har_models.create_1d_conv_core_model(input_shape)
    # core_model = mixer.create_mixer_core_model(
    #     input_shape=(300, 18),  # (T, C)
    #     patch_size=(9, 300),  # divides 18 and 300
    #     embed_dim=256,
    #     token_dim=128,
    #     channel_dim=512,
    #     num_blocks=6,
    #     dropout=0.1,
    #     use_rgb_embedding=True,  # keep True unless you want a pure 1-ch embed
    #     rgb_kernel_t=15,
    # )
    if experiment_type == "har_full_train" or experiment_type == 'har_full_fine_tune':
        full_model = self_har_models.attach_full_har_classification_head(core_model, output_shape, input_shape,
                                                                         optimizer=optimizer_instance,
                                                                         num_units=1024, model_name="HAR")
    elif experiment_type == "har_linear_train":
        full_model = self_har_models.attach_linear_classification_head(core_model, output_shape, input_shape,
                                                                       optimizer=optimizer_instance, model_name="Linear")

    elif experiment_type == "self_har":
        full_model = self_har_models.attach_multitask_transform_head(core_model,
                                                                     output_tasks=transform_funcs_names,
                                                                     input_shape=input_shape,
                                                                     optimizer=optimizer_instance,
                                                                     with_har_head=True,
                                                                     har_output_shape=output_shape,
                                                                     num_units_har=1024,
                                                                     model_name="StudentPreTrain"
                                                                     )



    else:
        raise ValueError(f"Unsupported model type: {experiment_type}")

    # Load the pretrained weights
    state_dict = torch.load(model_weights_path)
    model_keys = list(full_model['model'].state_dict().keys())  # Get correct layer names from model
    state_dict_keys = list(state_dict.keys())  # Get layer names from saved state_dict

    # Map state_dict keys to model keys in order
    new_state_dict = {model_key: state_dict[state_key] for model_key, state_key in zip(model_keys, state_dict_keys)}

    full_model['model'].load_state_dict(new_state_dict)

    # ✅ Move model to the appropriate device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_model['model'].to(device)

    # ✅ Set the model to evaluation mode
    full_model['model'].eval()

    print(f"Successfully loaded and prepared a '{experiment_type}' model for evaluation.")
    return full_model['model']

def optimizer_selection(optimizer_type, model=None, initial_learning_rate=0.0003):
    if optimizer_type.lower() == 'adam' and model == None:
        optimizer = torch.optim.Adam
    elif optimizer_type.lower() == 'sgd' and model == None:
        optimizer = torch.optim.SGD
    elif optimizer_type.lower() == 'adam' and model != None:
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    elif optimizer_type.lower() == 'sgd' and model != None:
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_learning_rate)

    return optimizer

# Set the configuration to default value if it was none
def get_config_default_value_if_none(experiment_config, entry, set_value=True):
    if entry in experiment_config:
        return experiment_config[entry]

    if entry == 'type':
        default_value = 'none'
    elif entry == 'tag':
        default_value = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    elif entry == 'previous_config_offset':
        default_value = 0
    elif entry == 'previous_config':
        default_value = 'none'
    elif entry == 'initial_learning_rate':
        default_value = 0.0003
    elif entry == 'epochs':
        default_value = 30
    elif entry == 'batch_size':
        default_value = 300
    elif entry == 'optimizer':
        default_value = 'adam'
    elif entry == 'self_training_samples_per_class':
        default_value = 10000
    elif entry == 'self_training_minimum_confidence':
        default_value = 0.0
    elif entry == 'self_training_plurality_only':
        default_value = True
    elif entry == 'trained_model_path':
        default_value = ''
    elif entry == 'trained_model_type':
        default_value = 'unknown'
    elif entry == 'eval_results':
        default_value = {}
    elif entry == 'eval_har':
        default_value = False
    elif entry == 'leave_one_subject_out_cross_validation':
        default_value = False
    elif entry == 'consistency_evaluation':
        default_value = False
    elif entry == 'testing_consistency_evaluation':
        default_value = False
    elif entry == 'probability_evaluation':
        default_value = False
    elif entry == 'consistency_over_time_evaluation':
        default_value = False
    elif entry == 'visualization':
        default_value = False
    elif entry == 'consistency_filter_steps':
        default_value = 2000
    elif entry == 'selection_method':
        default_value = 'none'
    elif entry == 'generate_pseudo_labels':
        default_value = False
    elif entry == "data_type":
        default_value = "source and target"
    elif entry == "coefficient_for_augmentation_loss":
        default_value = 1
    elif entry == "step":
        default_value = 10
    elif entry == "model_type":
        default_value = "None"
    elif entry == "percentage":
        default_value = True

    if set_value:
        experiment_config[entry] = default_value
        print(f"INFO: configuration {entry} set to default value: {default_value}.")

    return default_value

def get_parser(config="self_har", labelled_dataset_path="Pamap2_processed",
               unlabelled_dataset_path="Phone_hhar_processed", window_size=400, max_unlabelled_windows=40000):
    """
    This function, get_parser(), creates and configures an argparse.ArgumentParser object for handling command-line
    arguments in a Python script related to SelfHAR (Self Human Activity Recognition) training.

    The command-line arguments defined by this parser function are as follows:
    - --working_directory: Specifies the directory containing datasets, trained models, and training logs.
      Default value: 'run'.

    - --config: Allows specifying a configuration file for training methodology. The default value is
    'sample_configs/self_har.json'.

    - --labelled_dataset_path: Sets the path to the labeled dataset used for training and fine-tuning. This is a
    required parameter.

    - --unlabelled_dataset_path: Defines the path to the unlabelled dataset used for self-training and
    self-supervised training. It is ignored if only supervised training is performed. The default value is
    'run/processed_datasets/hhar_processed.pkl'.

    - --window_size: Specifies the size of the sliding window for data processing. The default value is 400.

    - --max_unlabelled_windows: Sets the maximum number of unlabelled windows. The default value is 40000,
    to avoid the long training time.

    - --use_tensor_board_logging: Allows enabling or disabling TensorBoard logging. The default value is True,
    and it is parsed as a boolean using the 'strtobool' function.

    - --verbose: Sets the verbosity level for the script. The default value is 1.

    Users can use these command-line arguments to customize the behavior of the SelfHAR training scفript,
    making it adaptable to various training scenarios.

    Usage: python train_selfhar.py --working_directory my_directory --labelled_dataset_path labeled_data.pkl
    --unlabelled_dataset_path unlabelled_data.pkl --window_size 500 --use_tensor_board_logging False --verbose 2

    Note: The purpose of the '--config' and '--max_unlabelled_windows' parameters is not explicitly described in this
    docstring and should be documented separately if needed.
    """

    def strtobool(v):
        return bool(distutils.util.strtobool(v))


    parser = argparse.ArgumentParser(
        description='SelfHAR Training')

    parser.add_argument('--working_directory', default='run',
                        help='directory containing datasets, trained models and training logs')
    parser.add_argument('--config', default=f'sample_configs/{config}.json',
                        help='')

    parser.add_argument('--labelled_dataset_path', default=f'run/processed_datasets/{labelled_dataset_path}.pkl',
                        type=str, help='name of the labelled dataset for training and fine-tuning')
    parser.add_argument('--unlabelled_dataset_path', default=f'run/processed_datasets/{unlabelled_dataset_path}.pkl',
                        type=str, help='name of the unlabelled dataset to self-training and self-supervised training, '
                             'ignored if only supervised training is performed.')

    parser.add_argument('--window_size', default=window_size, type=int,
                        help='the size of the sliding window')
    parser.add_argument('--max_unlabelled_windows', default=max_unlabelled_windows, type=int,  # 40000
                        help='')

    parser.add_argument('--use_tensor_board_logging', default=True, type=strtobool,
                        help='')
    parser.add_argument('--verbose', default=1, type=int,
                        help='verbosity level')

    return parser

def prepare_dataset(dataset_path, window_size, get_train_test_users, validation_split_proportion=0.1, verbose=1,
                    activity_mapping={}, label_list=None):
    if verbose > 0:
        print(f"Loading dataset at {dataset_path}")

    # Read dataset dictionary
    with open(dataset_path, 'rb') as f:
        dataset_dict = pickle.load(f)
        #user_datasets = dataset_dict['user_split']
        #label_list = dataset_dict['label_list']
        # Select the target activities and their data
        user_datasets, label_list = data_pre_processing.select_and_rename_target_activities(dataset_dict['user_split'],
                                                                                            activity_mapping,
                                                                                            verbose,
                                                                                            label_list=label_list,
                                                                                            )

    # Encode the label and set output shap
    label_map = dict([(l, i) for i, l in enumerate(label_list)])
    output_shape = len(label_list)

    # Split the users into training and testing users
    har_users = list(user_datasets.keys())
    train_users, test_users = get_train_test_users(har_users)
    if verbose > 0:
        print(f'Testing users: {test_users}, Training users: {train_users}')

    # Split the dataset into training, validation, and testing sets
    np_train, np_val, np_test = data_pre_processing.pre_process_dataset_composite( user_datasets=user_datasets,
                                                                                   label_map=label_map,
                                                                                   output_shape=output_shape,
                                                                                   train_users=train_users,
                                                                                   test_users=test_users,
                                                                                   window_size=window_size,
                                                                                   shift=window_size//2,
                                                                                   normalise_dataset=True,
                                                                                   validation_split_proportion=
                                                                                   validation_split_proportion,
                                                                                   verbose=verbose
                                                                                   )

    return {
        'train': np_train,
        'val': np_val,
        'test': np_test,
        'label_map': label_map,
        'input_shape': np_train[0].shape[1:],
        'output_shape': output_shape,
        'har_users': har_users,
    }

def generate_unlabelled_datasets_variations(unlabelled_data_x, labelled_data_x, labelled_repeat=1, verbose=1):
    """
    Generate variations of unlabelled datasets by combining unlabelled data with repeated labelled data.

    Parameters:
    - unlabelled_data_x: Unlabelled data to be combined.
    - labelled_data_x: Labelled data to be repeated and combined.
    - labelled_repeat: Number of times labelled data is repeated. Default is 1.
    - verbose: Verbosity level for printing information. Default is 1.

    Returns:
    - A dictionary with variations of unlabelled datasets.

    This function creates variations of unlabelled datasets by combining unlabelled data with repeated labelled data.
    It can be useful for generating training datasets for self-training tasks.

    Usage:
    - Provide unlabelled and labelled data along with optional parameters to generate variations.
    - Example: `variations = generate_unlabelled_datasets_variations(
            prepared_datasets['unlabelled'],
            prepared_datasets['labelled']['train'][0],
            labelled_repeat=labelled_repeat)

    Note:
    - The 'verbose' parameter controls whether information is printed during the process.
    """

    if verbose > 0:
        print("Unlabeled data shape: ", unlabelled_data_x.shape)

    # Repeat the labelled data
    labelled_data_repeat = np.repeat(labelled_data_x, labelled_repeat, axis=0)

    # Combine unlabelled and labelled data
    np_unlabelled_combined = np.concatenate([unlabelled_data_x, labelled_data_repeat])
    if verbose > 0:
        print(f"Unlabelled Combined shape: {np_unlabelled_combined.shape}")

    # Clean up memory
    gc.collect()

    return {
        'labelled_x_repeat': labelled_data_repeat,
        'unlabelled_combined': np_unlabelled_combined
    }

def load_unlabelled_dataset(prepared_datasets, unlabelled_dataset_path, window_size, labelled_repeat,
                            max_unlabelled_windows=None, verbose=1, activity_mapping={}, label_list=None):
    """
        Load unlabelled dataset, prepare it, and create variations for self-training.

        Parameters:
        - prepared_datasets: A dictionary containing prepared datasets.
        - unlabelled_dataset_path: File path to the unlabelled dataset.
        - window_size: The size of the sliding window for data processing.
        - labelled_repeat: Number of times labelled data is repeated when creating variations.
        - max_unlabelled_windows: Maximum number of unlabelled windows to retain.
        - verbose: Verbosity level for printing information. Default is 1.

        Returns:
        - Updated prepared_datasets dictionary with variations of datasets; labelled, unlabelled, labelled_x_repeat, and
        unlabelled_combined

        This function loads the unlabelled dataset, prepares it for training, and generates variations of unlabelled
        datasets  (unlabelled, labelled_x_repeat, and unlabelled_combined) for self-training tasks. It also allows you
        to limit the number of unlabelled windows if needed.

        Usage:
        - Provide the prepared datasets, unlabelled dataset file path, and other parameters to load and prepare the data.
        - Example: `prepared_datasets = load_unlabelled_dataset(prepared_datasets, args.unlabelled_dataset_path,
        window_size, labelled_repeat, max_unlabelled_windows=args.max_unlabelled_windows)


        Note:
        - The 'max_unlabelled_windows' parameter can be used to control the maximum number of unlabelled windows.
        """

    def get_empty_test_users(har_users):
        return (har_users, [])

    # Prepared the unlabelled dataset
    prepared_datasets['unlabelled'] = prepare_dataset(unlabelled_dataset_path, window_size, get_empty_test_users,
                                                      validation_split_proportion=0,
                                                      verbose=verbose,
                                                      activity_mapping=activity_mapping,
                                                      label_list=label_list,
                                                      )
                                                      #activity_mapping=activity_mapping)['train'][0]

    # Drop data beyond the maximum unlabelled windows if specified
    if max_unlabelled_windows is not None:
        #prepared_datasets['unlabelled'] = prepared_datasets['unlabelled'][:max_unlabelled_windows]
        prepared_datasets['unlabelled']['train'] = prepared_datasets['unlabelled']['train'][:max_unlabelled_windows]

    # Generate variations of unlabelled datasets (labelled_x_repeat, and unlabelled_combined)
    prepared_datasets = {
        **prepared_datasets,
        **generate_unlabelled_datasets_variations(#prepared_datasets['unlabelled'],
                                                  prepared_datasets['unlabelled']['train'][0],
                                                  prepared_datasets['labelled']['train'][0],
                                                  labelled_repeat=labelled_repeat
        )}
    return prepared_datasets

# Define a custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):  # Convert NumPy arrays to lists
            return obj.tolist()
        elif isinstance(obj, np.float64) or isinstance(obj, np.float32):  # Convert NumPy floats
            return float(obj)
        elif isinstance(obj, np.integer):  # Convert NumPy integers
            return int(obj)
        return super().default(obj)

def convert_experiment_configs(experiment_configs):
    """
    Converts NumPy objects in experiment_configs to Python-native types.
    """
    return json.loads(json.dumps(experiment_configs, cls=NumpyEncoder))


def to_one_hot(array):
    """
    Converts a 2D NumPy array of probabilities to one-hot encoded format.

    :param array: 2D NumPy array where each row represents class probabilities.
    :return: One-hot encoded array of the same shape.
    """
    one_hot = np.zeros_like(array)  # Initialize zero matrix of same shape
    max_indices = np.argmax(array, axis=1)  # Get index of max value for each row
    one_hot[np.arange(array.shape[0]), max_indices] = 1  # Set max index to 1
    return one_hot

def generate_seed(keyword: str) -> int:
    """Generates a reproducible but diverse integer seed from a keyword."""
    return int(hashlib.sha256(keyword.encode()).hexdigest(), 16) % (2**32)

