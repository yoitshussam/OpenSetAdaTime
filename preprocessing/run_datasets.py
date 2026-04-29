import os
import gc
import pickle
import argparse
import datetime
import requests
import zipfile
import copy
import distutils.util
import scipy.constants
import raw_data_processing
import re
from typing import Dict, Iterable, List, Optional
import pandas as pd

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

"""
Making use of the following datasets:
MotionSense
    @inproceedings{Malekzadeh:2019:MSD:3302505.3310068,
        author = {Malekzadeh, Mohammad and Clegg, Richard G. and Cavallaro, Andrea and Haddadi, Hamed},
        title = {Mobile Sensor Data Anonymization},
        booktitle = {Proceedings of the International Conference on Internet of Things Design and Implementation},
        series = {IoTDI '19},
        year = {2019},
        isbn = {978-1-4503-6283-2},
        location = {Montreal, Quebec, Canada},
        pages = {49--58},
        numpages = {10},
        url = {http://doi.acm.org/10.1145/3302505.3310068},
        doi = {10.1145/3302505.3310068},
        acmid = {3310068},
        publisher = {ACM},
        address = {New York, NY, USA},
        keywords = {adversarial training, deep learning, edge computing, sensor data privacy, time series analysis},
    }

HHAR Allan Stisen, Henrik Blunck, Sourav Bhattacharya, Thor Siiger Prentow, Mikkel Baun Kjærgaard, Anind Dey, 
Tobias Sonne, and Mads Møller Jensen "Smart Devices are Different: Assessing and Mitigating Mobile Sensing 
Heterogeneities for Activity Recognition" In Proc. 13th ACM Conference on Embedded Networked Sensor Systems (SenSys 
2015), Seoul, Korea, 2015. http://dx.doi.org/10.1145/2809695.2809718 """

DATASET_METADATA = {  # Used dataset information
    'Pamap2': {
        'name': 'Pamap2',
        'dataset_home_page': 'http://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring',
        'source_url': 'http://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip',
        'file_name': 'pamap2+physical+activity+monitoring.zip',

        'default_folder_path': 'pamap2+physical+activity+monitoring',
        'save_file_name': 'Pamap2_processed.pkl',
        'label_list': ['lying', 'sitting', 'standing', 'walking', 'running', 'cycling', 'Nordic walking', 'watching TV',
                       'computer work', 'car driving', 'ascending stairs', 'descending stairs', 'vacuum cleaning',
                       'ironing', 'folding laundry', 'house cleaning', 'playing soccer', 'rope jumping',
                       'other (transient activities)'],
        'label_list_full_name': ['lying', 'sitting', 'standing', 'walking', 'running', 'cycling', 'Nordic walking',
                                 'watching TV', ' computer work', 'car driving', 'ascending stairs',
                                 'descending stairs', 'vacuum cleaning', 'ironing', 'folding laundry', 'house cleaning',
                                 'playing soccer', 'rope jumping', 'other (transient activities)'],
        'has_null_class': True,
        'sampling_rate': 100.0,
        'unit': 1,
        'sensor_type': 'watch',
    },
    'MHEALTH': {
        'name': 'MHEALTH',
        'dataset_home_page': 'https://archive.ics.uci.edu/dataset/319/mhealth+dataset',
        'source_url': 'https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip',
        'file_name': 'mhealth+dataset.zip',

        'default_folder_path': 'mhealth+dataset',
        'save_file_name': 'mhealth_processed.pkl',
        'label_list': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],

        'label_list_full_name': ['null', 'Standing still', 'Sitting and relaxing', 'Lying down', 'Walking',
                                 'Climbing stairs', 'Waist bends forward', 'Frontal elevation of arms',
                                 'Knees bending (crouching)', 'Cycling', 'Jogging', 'Running', 'Jump front & back'],
        'has_null_class': True,
        'sampling_rate': 50.0,
        'unit': 1,
        'sensor_type': 'watch',
    },
    'RealWorld HAR': {
        'name': 'RealWorld HAR',
        'dataset_home_page': 'https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/',
        'source_url': 'http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip',
        'file_name': 'realworld2016_dataset.zip',

        'default_folder_path': 'realworld2016_dataset',
        'save_file_name': 'RealWorld_processed.pkl',
        'label_list': ['walking', 'running', 'sitting', 'standing', 'lying', 'climbingup', 'climbingdown', 'jumping'],
        'label_list_full_name': ['Walking', 'Running', 'Sitting', 'Standing', 'lying', 'Stairs Up', 'Stairs Down',
                                 'Jumping'],
        'has_null_class': False,
        'sampling_rate': 50.0,
        'unit': 1,
        'sensor_type': 'watch',
    },
    'motionsense': {
        'name': 'motionsense',
        'dataset_home_page': 'https://github.com/mmalekzadeh/motion-sense/',
        'source_url': 'https://github.com/mmalekzadeh/motion-sense/blob/master/data/B_Accelerometer_data.zip?raw=true',
        'file_name': 'B_Accelerometer_data.zip',

        'default_folder_path': 'B_Accelerometer_data',
        'save_file_name': 'motionsense_processed.pkl',
        'label_list': ['sit', 'std', 'wlk', 'ups', 'dws', 'jog'],
        'label_list_full_name': ['sitting', 'standing', 'walking', 'walking upstairs', 'walking downstairs', 'jogging'],
        'has_null_class': False,
        'sampling_rate': 50.0,
        'unit': scipy.constants.g,
        'sensor_type': 'phone',
    },
    'Watch_hhar': {
        'name': 'hhar',
        'dataset_home_page': 'http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition',
        'source_url': 'http://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip',
        'file_name': 'Activity recognition exp.zip',

        'default_folder_path': 'Activity recognition exp',
        'save_file_name': 'Watch_hhar_processed.pkl',
        'label_list': ['sit', 'stand', 'walk', 'stairsup', 'stairsdown', 'bike'],
        'label_list_full_name': ['sitting', 'standing', 'walking', 'walking upstairs', 'walking downstairs', 'biking'],
        'has_null_class': False,
        'sampling_rate': 200.0,
        'unit': 1,
        'sensor_type': 'watch',
    },
    'Phone_hhar': {
        'name': 'hhar',
        'dataset_home_page': 'http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition',
        'source_url': 'http://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip',
        'file_name': 'Activity recognition exp.zip',

        'default_folder_path': 'Activity recognition exp',
        'save_file_name': 'Phone_hhar_processed.pkl',
        'label_list': ['sit', 'stand', 'walk', 'stairsup', 'stairsdown', 'bike'],
        'label_list_full_name': ['sitting', 'standing', 'walking', 'walking upstairs', 'walking downstairs', 'biking'],
        'has_null_class': False,
        'sampling_rate': 100.0,
        'unit': 1,
        'sensor_type': 'phone',
    },
}
"""
    'Wisdm': {
        'name': 'Wisdm',
        'dataset_home_page': 'https://www.cis.fordham.edu/wisdm/dataset.php',
        'source_url': 'https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz',
        'file_name': 'WISDM_ar_latest.tar.gz',

        'default_folder_path': 'WISDM_ar_latest',
        'save_file_name': 'wisdm_processed.pkl',
        'label_list': ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing'],
        'label_list_full_name': ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing'],
        'has_null_class': False,
        'sampling_rate': 20.0,
        'unit': 1,
        'sensor_type': 'phone',
    },
    'ActiVAtE Student': {
        'name': 'ActiVAtE',
        'dataset_home_page': 'https://www.uni-vechta.de/activate',
        'source_url': '',
        'file_name': 'ActiVAtE Student.csv',

        'default_folder_path': '',
        'save_file_name': 'ActiVAtE_processed.pkl',
        'label_list': ['walking', 'running', 'sitting', 'lying'],
        'label_list_full_name': ['walking', 'running', 'sitting', 'lying'],
        'has_null_class': False,
        'sampling_rate': 13.0,
        'unit': 1,
        'sensor_type': 'watch',
    },
"""

ORIGINAL_DATASET_SUB_DIRECTORY = 'original_datasets'
PROCESSED_DATASET_SUB_DIRECTORY = 'processed_datasets'

"Downloading dataset general information"


def get_parser():
    """
    This function, get_parser(), creates and configures an argparse.ArgumentParser object for handling command-line
    arguments in a Python script related to SelfHAR (Self Human Activity Recognition) datasets download and processing.

    The command-line arguments defined by this parser function are as follows:
    - --working_directory: Specifies the output directory for downloaded and processed datasets.
      Default value: 'run'.

    - --mode: Sets the running mode of the script, with the following choices:
      - 'download_and_process': Downloads and processes the dataset(s).
      - 'download': Downloads the dataset(s).
      - 'process': Processes the downloaded dataset(s).
      Default value: 'download_and_process'.

    - --dataset: Defines the name of the dataset to be downloaded or processed, with the following choices:
      - 'motionsense': Refers to the MotionSense dataset.
      - 'hhar': Refers to the HHAR dataset.
      - 'all': Indicates all available datasets can be selected.
      Default value: 'all'.

    - --dataset_file_path: Specifies the path to the downloaded dataset for processing. If not provided, the default download path is used.
      Default value: None.

    """

    parser = argparse.ArgumentParser(
        description='SelfHAR datasets download and processing script')
    parser.add_argument('--working_directory', default='run',
                        help='the output directory of the downloads and processed datasets')
    # parser.add_argument('--mode', default='process',
    parser.add_argument('--mode', default='process',
                        choices=['download_and_process', 'download', 'process'],
                        help='the running mode of the script.\ndownload: download the dataset(s).\nprocess: process the downloaded dataset(s)')
    parser.add_argument('--dataset', default='all',
                        choices=['motionsense', 'hhar', 'Wisdm', 'all'],
                        help='name of the dataset to be downloaded/processed')
    parser.add_argument('--dataset_file_path', default=None,
                        help='the path to the downloaded dataset for processing. Default download path is used when None.')
    return parser


def download_dataset(data_directory, dataset_metadata):
    message = f"""You are going to download the '{dataset_metadata['name']}' dataset.
    Please verify that you have visited the homepage of the dataset
    (link: {dataset_metadata['dataset_home_page']} . Note: this link is not necessarily up-to-date or accurate),
    and read any other document accompanying this dataset,
    and agree to all the terms and conditions set out by the dataset authors/data collectors.
    The author of this script is not liable for any use of this script, or any use or download of the dataset.
    You agree to be the person responsible for the download and any subsequent use of the dataset.
    Please enter 'y' to agree to the terms above, in addition to any other terms previously set out.
    """
    # answer = distutils.util.strtobool(input(message))
    answer = input(message)  # Ask for permission to download the dataset
    if answer == 'y':
        dataset_name = dataset_metadata['name']
        dataset_url = dataset_metadata['source_url']
        file_name = dataset_metadata['file_name']

        if not os.path.exists(os.path.join(data_directory, dataset_name)):  # Check if the dataset directory exists
            os.mkdir(os.path.join(data_directory, dataset_name))

        print("Downloading ...")  # Download the dataset
        r = requests.get(dataset_url, allow_redirects=True)
        with open(os.path.join(data_directory, dataset_name, file_name), 'wb') as f:
            f.write(r.content)
        print(f"Finished downloading to ({os.path.join(data_directory, dataset_name, file_name)})")
    else:
        print("You did not agree to the terms.")


def split_dataset_by_sensor_groups_and_save(
    dataset_name: str,
    group1_name: str,
    group2_name: str,
    processed_dataset_directory: str,
    dataset_metadata: dict,
    dataset_folder_path: str,
    new_sampling_rate: int,
    resampling: bool,
):
    """
    Create two processed datasets by sensor-group (e.g., hand vs leg) and save them as .pkl
    EXACTLY like split_dataset_by_subject_groups_and_save() does. :contentReference[oaicite:1]{index=1}

    Output files go into processed_dataset_directory with names:
      {prefix}_{group}_processed.pkl
    Example:
      RealWorld_hand_processed.pkl
      RealWorld_leg_processed.pkl
    """

    # 1) Map group names -> sensor_position labels
    sensor_group_map = {
        "hand": ["forearm", "hand", "right_lower_arm"],
        "leg": ["shin", "ankle", "left_ankle"],
    }

    allowed_positions = ["forearm", "shin", "hand", "ankle", "left_ankle", "right_lower_arm"]

    def _resolve_positions(group_name: str):
        g = (group_name or "").strip().lower()
        if g in sensor_group_map:
            return sensor_group_map[g]
        if g in allowed_positions:
            return [g]
        raise ValueError(f"Unknown sensor group '{group_name}'. Use 'hand' or 'leg' (or one allowed position).")

    # 2) Pick the correct processing function by dataset
    if dataset_name == "RealWorld HAR":
        process_fn = raw_data_processing.process_RealWorld_HAR_files
    elif dataset_name == "MHEALTH":
        process_fn = raw_data_processing.process_MHEALTH_files
    elif dataset_name == "Pamap2":
        process_fn = raw_data_processing.process_Pamap2_files
    else:
        raise ValueError(f"Unsupported dataset_name='{dataset_name}' for sensor split.")

    os.makedirs(processed_dataset_directory, exist_ok=True)

    # ---- Build metadata prefix exactly like your subject-split function ---- :contentReference[oaicite:2]{index=2}
    base_save_name = dataset_metadata.get("save_file_name", f"{dataset_name}_processed.pkl")
    base_name, ext = os.path.splitext(base_save_name)
    if ext == "":
        ext = ".pkl"

    if base_name.endswith("_processed"):
        prefix = base_name[:-len("_processed")]
    else:
        prefix = base_name

    def _process_and_save_one_group(group_name: str):
        sensor_position = _resolve_positions(group_name)

        processed_user_split = process_fn(
            data_folder_path=dataset_folder_path,
            sensor_type=dataset_metadata["sensor_type"],
            old_sampling_rate=dataset_metadata["sampling_rate"],
            new_sampling_rate=new_sampling_rate,
            resampling=resampling,
            sensor_position=sensor_position,
        )

        # ---- Save EXACTLY like split_dataset_by_subject_groups_and_save ---- :contentReference[oaicite:3]{index=3}
        group_metadata = copy.copy(dataset_metadata)
        group_metadata["name"] = f"{dataset_name}_{group_name}"
        group_metadata["save_file_name"] = f"{prefix}_{group_name}_processed{ext}"

        group_content = copy.copy(group_metadata)
        group_content["user_split"] = processed_user_split

        group_path = os.path.join(processed_dataset_directory, group_metadata["save_file_name"])
        with open(group_path, "wb") as f:
            pickle.dump(group_content, f)

        print(
            f"Saved sensor-group '{group_name}' to {group_path} "
            f"(sensor_position={sensor_position})."
        )
        return group_path

    group1_path = _process_and_save_one_group(group1_name)
    group2_path = _process_and_save_one_group(group2_name)

    return group1_path, group2_path


def split_dataset_by_subject_groups_and_save(
    dataset_name: str,
    user_datasets: dict,
    group1_ids,
    group2_ids,
    group1_name: str = "group1",
    group2_name: str = "group2",
    processed_dataset_directory: str = ".",
    dataset_metadata: dict | None = None,
):
    """
    Split a dataset by two subject groups and save two new processed dataset files.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., "RealWorld HAR", "Pamap2").
        Used only for metadata 'name' field and logging.

    user_datasets : dict
        Mapping from subject identifier -> processed data.
        Subject identifiers can be strings like "proband8", "subject_10", or integers.

    group1_ids : iterable
        Subject IDs belonging to the first group.
        Can be:
            - numeric IDs (e.g., {2, 3, 4})
            - or exact keys (e.g., {"proband2", "proband3"}).

    group2_ids : iterable
        Subject IDs belonging to the second group (same format as group1_ids).

    group1_name : str, optional
        Name of the first group (used in file names).
        Default: "group1"

    group2_name : str, optional
        Name of the second group (used in file names).
        Default: "group2"

    processed_dataset_directory : str
        Directory where the new processed .pkl files will be saved.

    dataset_metadata : dict, optional
        Base metadata dictionary for the dataset (as used in your main processing).
        Must contain 'save_file_name'. If None, minimal metadata is built.
    """

    # Normalize group IDs: store both as strings and ints for robust matching
    group1_ids_int = set()
    group1_ids_str = set()
    for gid in group1_ids:
        group1_ids_str.add(str(gid))
        try:
            group1_ids_int.add(int(gid))
        except (TypeError, ValueError):
            pass

    group2_ids_int = set()
    group2_ids_str = set()
    for gid in group2_ids:
        group2_ids_str.add(str(gid))
        try:
            group2_ids_int.add(int(gid))
        except (TypeError, ValueError):
            pass

    group1_user_datasets = {}
    group2_user_datasets = {}

    for subj_key, subj_data in user_datasets.items():
        key_str = str(subj_key)

        # 1) Try direct string membership
        if key_str in group1_ids_str:
            group1_user_datasets[subj_key] = subj_data
            continue
        if key_str in group2_ids_str:
            group2_user_datasets[subj_key] = subj_data
            continue

        # 2) Otherwise extract numeric part from key (e.g. "proband8" -> 8)
        digits = re.sub(r"\D+", "", key_str)
        sid_int = None
        if digits != "":
            try:
                sid_int = int(digits)
            except ValueError:
                sid_int = None

        if sid_int is not None:
            if sid_int in group1_ids_int:
                group1_user_datasets[subj_key] = subj_data
            elif sid_int in group2_ids_int:
                group2_user_datasets[subj_key] = subj_data
            else:
                print(f"Warning: subject {sid_int} (key '{subj_key}') not in group1 or group2; skipping.")
        else:
            print(f"Warning: could not extract numeric ID from key '{subj_key}'; skipping.")

    # ---------- Build metadata and filenames ----------
    if dataset_metadata is None:
        dataset_metadata = {
            "name": dataset_name,
            "save_file_name": f"{dataset_name}_processed.pkl",
        }

    base_save_name = dataset_metadata.get("save_file_name", f"{dataset_name}_processed.pkl")
    base_name, ext = os.path.splitext(base_save_name)
    if ext == "":
        ext = ".pkl"

    # Try to strip trailing "_processed" to get a cleaner prefix (e.g., RealWorld_processed -> RealWorld)
    if base_name.endswith("_processed"):
        prefix = base_name[:-len("_processed")]
    else:
        prefix = base_name

    # ---------- Group 1 save ----------
    group1_metadata = copy.copy(dataset_metadata)
    group1_metadata["name"] = f"{dataset_name}_{group1_name}"
    group1_metadata["save_file_name"] = f"{prefix}_{group1_name}_processed{ext}"

    group1_content = copy.copy(group1_metadata)
    group1_content["user_split"] = group1_user_datasets

    group1_path = os.path.join(processed_dataset_directory, group1_metadata["save_file_name"])
    with open(group1_path, "wb") as f:
        pickle.dump(group1_content, f)

    print(f"Saved group '{group1_name}' ({len(group1_user_datasets)} subjects) to {group1_path}.")

    # ---------- Group 2 save ----------
    group2_metadata = copy.copy(dataset_metadata)
    group2_metadata["name"] = f"{dataset_name}_{group2_name}"
    group2_metadata["save_file_name"] = f"{prefix}_{group2_name}_processed{ext}"

    group2_content = copy.copy(group2_metadata)
    group2_content["user_split"] = group2_user_datasets

    group2_path = os.path.join(processed_dataset_directory, group2_metadata["save_file_name"])
    with open(group2_path, "wb") as f:
        pickle.dump(group2_content, f)

    print(f"Saved group '{group2_name}' ({len(group2_user_datasets)} subjects) to {group2_path}.")

    return group1_path, group2_path

def process_dataset(data_directory, processed_dataset_directory, dataset_metadata, dataset_file_path=None):
    """
    This function, process_dataset(), is responsible for unzipping and processing a dataset. It takes several parameters to perform these tasks.

    Parameters:
    - data_directory: The directory where the original dataset and processed data will be stored.
    - processed_dataset_directory: The directory where the processed dataset will be saved.
    - dataset_metadata: A dictionary containing metadata about the dataset, including its name, file name, default folder path, and activities.
    - dataset_file_path (optional): The path to the downloaded dataset file, if it is not located in the default location.

    The function performs the following steps:
    1. Unzips the dataset by extracting its contents to the specified data_directory.
    2. Processes the dataset by calling a processing function based on the dataset's name.
       - If the dataset is 'hhar', it uses the 'process_hhar_accelerometer_files' function.
       - If the dataset is 'motionsense', it uses the 'process_motion_sense_accelerometer_files' function.
       - If the dataset is not supported, it prints an error message and returns.
    3. Creates a dictionary called 'dataset_content' that includes the dataset metadata and the processed data.
    4. Saves 'dataset_content' to a pkl file in the processed_dataset_directory.

    The function provides an informative message upon successful completion, indicating where the processed dataset has been saved.

    Usage:
    process_dataset(data_directory, processed_dataset_directory, dataset_metadata, dataset_file_path=None)

    Note: To add support for a new dataset, create a processing function in the 'raw_data_processing' module and add it to the 'args.dataset' dictionary.
    """

    dataset_name = dataset_metadata['name']
    file_name = dataset_metadata['file_name']
    #new_sampling_rate = 13.0
    new_sampling_rate = 50.0
    resampling = True

    print("Unzipping dataset...")
    dataset_file_path = args.dataset_file_path
    if dataset_file_path is None:  # Set dataset path
        dataset_file_path = os.path.join(data_directory, dataset_name, file_name)

    if dataset_name != "ActiVAtE":
        with zipfile.ZipFile(dataset_file_path, 'r') as zip_ref:  # Unzipping dataset
            zip_ref.extractall(os.path.join(data_directory, dataset_name))

    print("Processing dataset...")
    # To add a new dataset just creat a processing funtion in the raw_data_processing and add it to the args dataset
    dataset_folder_path = os.path.join(data_directory, dataset_name, dataset_metadata['default_folder_path'])
    # print("PATH", dataset_folder_path)
    if dataset_name == 'hhar':  # Check if the dataset is hhar
        # Process the harr dataset
        user_datasets = raw_data_processing.process_hhar_accelerometer_files(data_folder_path=dataset_folder_path,
                                                                             sensor_type=dataset_metadata[
                                                                                 'sensor_type'],
                                                                             old_sampling_rate=dataset_metadata[
                                                                                 'sampling_rate'],
                                                                             new_sampling_rate=new_sampling_rate,
                                                                             resampling=resampling)
    elif dataset_name == 'ActiVAtE':  # Check if the dataset is ActiVAtE
        # Process the harr dataset
        user_datasets = raw_data_processing.process_ActiVAtE_accelerometer_files(data_folder_path=dataset_folder_path,
                                                                             sensor_type=dataset_metadata[
                                                                                 'sensor_type'],
                                                                             old_sampling_rate=dataset_metadata[
                                                                                 'sampling_rate'],
                                                                             new_sampling_rate=new_sampling_rate,
                                                                             resampling=resampling)
    elif dataset_name == 'motionsense':  # Check if the dataset is motionsense
        # Process the motionsense dataset
        user_datasets = raw_data_processing.process_motion_sense_accelerometer_files(
            data_folder_path=dataset_folder_path,
            sensor_type=dataset_metadata['sensor_type'],
            old_sampling_rate=dataset_metadata['sampling_rate'],
            new_sampling_rate=new_sampling_rate,
            resampling=resampling)
    elif dataset_name == 'Wisdm':  # Check if the dataset is Wisdm
        # Process the Wisdm dataset
        user_datasets = raw_data_processing.process_Wisdm_accelerometer_files(data_folder_path=dataset_folder_path,
                                                                              sensor_type=dataset_metadata[
                                                                                  'sensor_type'],
                                                                              old_sampling_rate=dataset_metadata[
                                                                                  'sampling_rate'],
                                                                              new_sampling_rate=new_sampling_rate,
                                                                              resampling=resampling)
    elif dataset_name == 'Pamap2':  # Check if the dataset is Pamap2
        # Process the Pamap2 dataset
        user_datasets = raw_data_processing.process_Pamap2_files(data_folder_path=dataset_folder_path,
                                                                 sensor_type=dataset_metadata['sensor_type'],
                                                                 old_sampling_rate=dataset_metadata['sampling_rate'],
                                                                 new_sampling_rate=new_sampling_rate,
                                                                 resampling=resampling)
        split_dataset_by_sensor_groups_and_save(
            dataset_name=dataset_name,
            group1_name="hand",
            group2_name="leg",
            processed_dataset_directory=processed_dataset_directory,
            dataset_metadata=dataset_metadata,
            dataset_folder_path=dataset_folder_path,
            new_sampling_rate=new_sampling_rate,
            resampling=resampling,
        )

    elif dataset_name == 'MHEALTH':  # Check if the dataset is MHEALTH
        # Process the MHEALTH dataset
        # user_datasets = raw_data_processing.process_MHEALTH_accelerometer_files(data_folder_path=dataset_folder_path,
        user_datasets = raw_data_processing.process_MHEALTH_files(data_folder_path=dataset_folder_path,
                                                                  sensor_type=dataset_metadata['sensor_type'],
                                                                  old_sampling_rate=dataset_metadata['sampling_rate'],
                                                                  new_sampling_rate=new_sampling_rate,
                                                                  resampling=resampling)
        split_dataset_by_sensor_groups_and_save(
            dataset_name=dataset_name,
            group1_name="hand",
            group2_name="leg",
            processed_dataset_directory=processed_dataset_directory,
            dataset_metadata=dataset_metadata,
            dataset_folder_path=dataset_folder_path,
            new_sampling_rate=new_sampling_rate,
            resampling=resampling,
        )

    elif dataset_name == 'RealWorld HAR':  # Check if the dataset is RealWorld HAR
        # Process the RealWorld HAR dataset
        # user_datasets = raw_data_processing.process_RealWorld_HAR_accelerometer_files(
        user_datasets = raw_data_processing.process_RealWorld_HAR_files(data_folder_path=dataset_folder_path,
                                                                        sensor_type=dataset_metadata['sensor_type'],
                                                                        old_sampling_rate=dataset_metadata['sampling_rate'],
                                                                        new_sampling_rate=new_sampling_rate,
                                                                        resampling=resampling)
        split_dataset_by_sensor_groups_and_save(
            dataset_name=dataset_name,
            group1_name="hand",
            group2_name="leg",
            processed_dataset_directory=processed_dataset_directory,
            dataset_metadata=dataset_metadata,
            dataset_folder_path=dataset_folder_path,
            new_sampling_rate=new_sampling_rate,
            resampling=resampling,
        )

    else:  # Otherwise, the dataset is not supported
        print(f"Dataset {dataset_name} is not supported.")
        return
    # Create a dataset content and add the user datasets, which got from the raw_data_processing, to it
    dataset_content = copy.copy(dataset_metadata)
    dataset_content['user_split'] = user_datasets
    # Save the dataset content to the run\\processed_datasets as pkl file
    with open(os.path.join(processed_dataset_directory, dataset_metadata['save_file_name']), 'wb') as f:
        pickle.dump(dataset_content, f)
    print(f"Finished Processing, saved to "
          f"{os.path.join(processed_dataset_directory, dataset_metadata['save_file_name'])}.")

    if dataset_name == "RealWorld HAR":
        male_ids = {"proband2", "proband3", "proband4", "proband5", "proband7", "proband9", "proband10", "proband14",}
        female_ids = {"proband1", "proband6", "proband8", "proband11", "proband12", "proband13", "proband15",}

        split_dataset_by_subject_groups_and_save(
            dataset_name=dataset_name,
            user_datasets=user_datasets,
            group1_ids=male_ids,
            group2_ids=female_ids,
            group1_name="male",
            group2_name="female",
            processed_dataset_directory=processed_dataset_directory,
            dataset_metadata=dataset_metadata,
        )

    elif dataset_name == "Pamap2":
        male_ids = {"subject101", "subject103", "subject104", "subject105", "subject106", "subject107", "subject108", "subject109"}
        female_ids = {"subject102"}

        split_dataset_by_subject_groups_and_save(
            dataset_name=dataset_name,
            user_datasets=user_datasets,
            group1_ids=male_ids,
            group2_ids=female_ids,
            group1_name="male",
            group2_name="female",
            processed_dataset_directory=processed_dataset_directory,
            dataset_metadata=dataset_metadata,
        )

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists(args.working_directory):  # Check if the working directory exists
        os.mkdir(args.working_directory)
    dataset_directory = os.path.join(args.working_directory, ORIGINAL_DATASET_SUB_DIRECTORY)
    if not os.path.exists(dataset_directory):  # Check if the original datasets' directory exists
        os.mkdir(dataset_directory)

    processed_dataset_directory = os.path.join(args.working_directory, PROCESSED_DATASET_SUB_DIRECTORY)
    if not os.path.exists(processed_dataset_directory):  # Check if the processed datasets' directory exists
        os.mkdir(processed_dataset_directory)

    if args.mode == 'download_and_process' or args.mode == 'download':  # Download the datasets
        if args.dataset == 'all':  # Check which datasets to download
            datasets = list(DATASET_METADATA.keys())
        else:
            datasets = [args.dataset]

        for dataset in datasets:  # Download the datasets
            print(f"-------- Downloading {dataset} --------")
            download_dataset(dataset_directory, DATASET_METADATA[dataset])

    if args.mode == 'download_and_process' or args.mode == 'process':  # Datasets processing
        if args.dataset == 'all':  # Check which  datasets to process
            datasets = list(DATASET_METADATA.keys())
        else:
            datasets = [args.dataset]

        for dataset in datasets:
            print(f"-------- Processing {dataset} --------")
            process_dataset(dataset_directory, processed_dataset_directory, DATASET_METADATA[dataset],
                            args.dataset_file_path)
    print("Finished")
