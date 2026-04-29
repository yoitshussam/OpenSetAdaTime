import glob
import re
import os
import pandas as pd
import numpy as np
import zipfile
import re
from data_resampling import resample_data
from scipy.interpolate import interp1d

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

def unzip_file(zip_file_path):

    if not os.path.exists(zip_file_path):
        print(f"The specified ZIP file '{zip_file_path}' does not exist.")
        return
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        folder_path = os.path.dirname(zip_file_path)
        zip_file.extractall(folder_path)

def keep_cols(df, sensor_position=None, sensor=None, axes=None):
    # defaults → everything
    if sensor_position is None:
        # sensor_position = ['forearm', 'hand']
        # sensor_position = ['shin', 'ankle']
        sensor_position = ['forearm', 'shin', 'hand', 'ankle', 'left_ankle', 'right_lower_arm']
    if sensor is None:
        sensor = ['acc', 'gyr', 'mag', 'gyro', 'magnet']
        # sensor = ['acc']
    if axes is None:
        axes = ['x', 'y', 'z']

    # build the list of axis‐columns you want, e.g. 'acc_forearm_x', etc.
    cols = [
        f"{meas}_{ax}_{pos}"
        for pos in sensor_position
        for meas in sensor
        for ax in axes
    ]
    # keep only those that actually exist
    cols = [c for c in cols if c in df.columns]
    return cols

def get_user_datasets(resampled_dataset, user, sensor_position=None, sensor=None, axes=None):
    """
    sensor_position: list of body parts, e.g. ['forearm','shin']
    sensor:          list of sensor types, e.g. ['acc','gyr','mag']
    axes:            list of axes, e.g. ['x','y','z']
    """

    cols = keep_cols(resampled_dataset, sensor_position=sensor_position)

    # always include user-id and label
    keep = ['user-id', 'label'] + cols

    user_extract = resampled_dataset[resampled_dataset["user-id"] == user]
    data = user_extract[cols].values
    labels = user_extract["label"].values
    print(f"{user} {data.shape}")
    return [(data, labels)]

def process_motion_sense_accelerometer_files(data_folder_path, sensor_type='watch', old_sampling_rate=50, new_sampling_rate=100, resampling=False):
    """
    Preprocess the accelerometer files of the MotionSense dataset into the 'user-list' format
    Data files can be found at https://github.com/mmalekzadeh/motion-sense/tree/master/data
    Parameters:
        data_folder_path (str):
            the path to the folder containing the data files (unzipped)
            e.g. motionSense/B_Accelerometer_data/
            the trial folders should be directly inside it (e.g. motionSense/B_Accelerometer_data/dws_1/)
    Return:
        
        user_datsets (dict of {user_id: [(sensor_values, activity_labels)]})
            the processed dataset in a dictionary, of type {user_id: [(sensor_values, activity_labels)]}
            the keys of the dictionary is the user_id (participant id)
            the values of the dictionary are lists of (sensor_values, activity_labels) pairs
                sensor_values are 2D numpy array of shape (length, channels=3)
                activity_labels are 1D numpy array of shape (length)
                each pair corresponds to a separate trial 
                    (i.e. time is not contiguous between pairs, which is useful for making sliding windows, where it is easy to separate trials)
    """

    # label_set = {}
    user_datasets = {}
    all_trials_folders = sorted(glob.glob(data_folder_path + "/*"))  # Get all folders in the main folder

    # Loop through every trial folder
    for trial_folder in all_trials_folders:
        trial_name = os.path.split(trial_folder)[-1]  # Get trail name which represent the activity name

        # label of the trial is given in the folder name, separated by underscore
        label = trial_name.split("_")[0]
        # label_set[label] = True
        print(trial_folder)

        # Loop through files for every user of the trail
        for trial_user_file in sorted(glob.glob(trial_folder + "/*.csv")):

            # use regex to match the user id
            user_id_match = re.search(r'(?P<user_id>[0-9]+)\.csv', os.path.split(trial_user_file)[-1])
            if user_id_match is not None:
                user_id = int(user_id_match.group('user_id'))

                # Read file
                user_trial_dataset = pd.read_csv(trial_user_file)
                # user_trial_dataset.dropna(how="any", inplace=True)  # Drop the nan
                user_trial_dataset = user_trial_dataset.interpolate()  # Interpolate the nan

                user_trial_dataset = user_trial_dataset.drop(columns=["Unnamed: 0"])
                user_trial_dataset.columns = ["x-axis", "y-axis", "z-axis"]

                # Resample the dataset
                if resampling:
                    user_trial_dataset = resample_data(user_trial_dataset, old_sampling_rate, new_sampling_rate)

                # Extract the x, y, z channels
                values = user_trial_dataset[["x-axis", "y-axis", "z-axis"]].values

                # the label is the same during the entire trial, so it is repeated here to pad to the same length as the values
                labels = np.repeat(label, values.shape[0])

                if user_id not in user_datasets:
                    user_datasets[user_id] = []
                user_datasets[user_id].append((values, labels))  # append the values and labels to user datasets
            else:
                print("[ERR] User id not found", trial_user_file)

    return user_datasets


def process_hhar_accelerometer_files(data_folder_path, sensor_type='watch', old_sampling_rate=200, new_sampling_rate=100, resampling=False):
    """
    Preprocess the accelerometer files of the HHAR dataset into the 'user-list' format
    Data files can be found at http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition

    """
    # print(data_folder_path)
    if sensor_type == 'phone':
        # Read the phones' accelerometer data
        har_dataset = pd.read_csv(os.path.join(data_folder_path, 'Phones_accelerometer.csv'))
    else:
        # Read the watches' accelerometer data
        har_dataset = pd.read_csv(os.path.join(data_folder_path, 'Watch_accelerometer.csv'))
        #har_dataset = har_dataset.loc[har_dataset["Device"] == "gear_1"].reset_index(drop=True)

    har_dataset['Arrival_Time'] = pd.to_datetime(har_dataset['Arrival_Time'], unit='ms')
    #print(f"{sensor_type} hhar sampling frequency is {1 / har_dataset['Arrival_Time'].diff().median().total_seconds()} Hz")

    #har_dataset.dropna(how="any", inplace=True)
    har_dataset = har_dataset[["Arrival_Time", "x", "y", "z", "gt", "User"]]
    har_dataset.columns = ["time", "x-axis", "y-axis", "z-axis", "label", "user-id"]
    har_users = har_dataset["user-id"].unique()

    # Interpolate the nan
    har_dataset[["x-axis", "y-axis", "z-axis"]] = har_dataset[["x-axis", "y-axis", "z-axis"]].interpolate()
    har_dataset[["user-id", 'label']] = har_dataset[["user-id", 'label']].ffill()

    # Resample the dataset
    if resampling:
        # resampled_dataset = resample_data(har_dataset[har_dataset["user-id"] == user], old_sampling_rate, new_sampling_rate)
        resampled_dataset = resample_data(har_dataset, old_sampling_rate, new_sampling_rate)
    else:
        resampled_dataset = har_dataset
    # Create the user datasets to save it as pkl file
    user_datasets = {}
    for user in har_users:

        user_datasets[user] = get_user_datasets(resampled_dataset, user)

    return user_datasets


def process_Wisdm_accelerometer_files(data_folder_path, sensor_type='watch', old_sampling_rate=20, new_sampling_rate=100, resampling=False):
    columns = ['user-id', 'label', 'time', "x-axis", "y-axis", "z-axis"]
    # df = pd.read_csv(data_folder_path, header = None, delim_whitespace=True, names=columns)
    data = pd.read_csv(data_folder_path, header=None, delim_whitespace=True)
    data = data[0].str.split(';', 1, expand=True)
    data = data[0].str.split(',', len(columns) - 1, expand=True)
    data.columns = columns
    # data["user-id"] = data["user-id"].astype(int)
    data["user-id"] = data["user-id"].astype(str)
    data['time'] = data['time'].astype(float)
    axis = ["x-axis", "y-axis", "z-axis"]
    data[axis] = data[axis].apply(pd.to_numeric, errors='coerce', axis=1)
    data["time"] = pd.to_datetime(data["time"], unit='ns')
    # DataBase["time"] = unix(DataBase['time'])
    # pd.to_datetime(DataBase["time"][0], unit='ns')
    data = data.dropna().reset_index(drop=True)

    if resampling:
        # resampled_dataset = resample_data(har_dataset[har_dataset["user-id"] == user], old_sampling_rate, new_sampling_rate)
        resampled_dataset = resample_data(data, old_sampling_rate, new_sampling_rate)
    """
    if resampling is True:
        data_base = pd.DataFrame()
        for user_id in data['user-id'].unique():
            data = data.loc[data['user-id'] == user_id]

            data = Downsampling(data, new_sampling_rate=13, old_sampling_rate=F)
            data_base = pd.concat([data_base, data])
    else:
        data_base = data
    """
    return resampled_dataset


def name_pamap2_columns(df):
    # 1) The first three columns
    cols = ['time', 'label', 'heart_rate']

    # 2) The 3 IMU blocks in order: hand, chest, ankle
    imu_positions = ['hand', 'chest', 'ankle']
    # within each block, these are the 17 fields—note we drop the '16' for the ±16 g accel
    imu_fields = [
        'temperature',
        # ±16 g accel (now named without the 16)
        'acc_x', 'acc_y', 'acc_z',
        # ±6 g accel
        'acc6_x', 'acc6_y', 'acc6_z',
        # gyro
        'gyr_x', 'gyr_y', 'gyr_z',
        # magnetometer
        'mag_x', 'mag_y', 'mag_z',
        # orientation (invalid in this dataset)
        'orient1', 'orient2', 'orient3', 'orient4'
    ]

    for pos in imu_positions:
        cols += [f"{fld}_{pos}" for fld in imu_fields]

    # Trim to the actual number of columns in df
    df.columns = cols[:df.shape[1]]
    return df


def process_Pamap2_files(data_folder_path, sensor_type='watch', old_sampling_rate=100, new_sampling_rate=100,
                         resampling=False, sensor_position=None, axis=[4, 5, 6, 10, 11, 12, 13, 14, 15]):
                                       # axis=[4, 5, 6]):

    # Activities list
    activity_list = ["other (transient activities)", "lying", "sitting", "standing", "walking", "running", "cycling",
                     "Nordic walking", "ascending stairs", "descending stairs", "vacuum cleaning", "ironing",
                     "rope jumping"]

    # Unzip the pamap2 dataset
    data_folder_path = data_folder_path.replace("pamap2+physical+activity+monitoring", 'PAMAP2_Dataset')
    zip_file_path = f'{data_folder_path}.zip'
    unzip_file(zip_file_path)

    # Read datasets' names in dataset files
    dataset_files = sorted(glob.glob(f'{data_folder_path}/Protocol' + "/*.dat"))

    # Get users_id
    users_id = [f"subject{match.group(1)}" for match in (re.search(r'subject(\d+)', dataset_file) for dataset_file
                                                         in dataset_files) if match]
    data_base = pd.DataFrame()

    # Create the user datasets to save it as pkl file
    user_datasets = {}

    for user in users_id:

        # Read user data
        matching = [s for s in dataset_files if user in s]
        data = pd.read_csv(matching[0], header=None, delim_whitespace=True)


        # Rename the columns
        data = name_pamap2_columns(data)
        data.insert(0, "user-id", user)
        # data = data.rename(columns={0: "time"})
        # data = data.rename(columns={1: "label"})
        # 1) Rename accelerometer axes
        # data = data.rename(columns={
        #     axis[0]: 'acc-x',
        #     axis[1]: 'acc-y',
        #     axis[2]: 'acc-z',
        # })

        # 2) Rename gyroscope axes
        # data = data.rename(columns={
        #     axis[3]: 'gyr-x',
        #     axis[4]: 'gyr-y',
        #     axis[5]: 'gyr-z',
        # })

        # 3) Rename magnetometer axes
        # data = data.rename(columns={
        #     axis[6]: 'mag-x',
        #     axis[7]: 'mag-y',
        #     axis[8]: 'mag-z',
        # })

        # 4) Select only the columns you need
        #    (assuming 'time', 'label', 'user-id' already exist)
        # data = data[['time',
        #              'label',
        #              'acc-x', 'acc-y', 'acc-z',
        #              'gyr-x', 'gyr-y', 'gyr-z',
        #              'mag-x', 'mag-y', 'mag-z',
        #              'user-id']]

        # data = data.rename(columns={axis[0]: "x-axis", axis[1]: "y-axis", axis[2]: "z-axis"})
        # data = data[['time', 'label', "x-axis", "y-axis", "z-axis", "user-id"]]

        # Interpolate the nan
        # data[["x-axis", "y-axis", "z-axis"]] = data[["x-axis", "y-axis", "z-axis"]].interpolate()
        # data[['acc-x', 'acc-y', 'acc-z', 'gyr-x', 'gyr-y', 'gyr-z', 'mag-x', 'mag-y', 'mag-z']] =\
        #     data[['acc-x', 'acc-y', 'acc-z', 'gyr-x', 'gyr-y', 'gyr-z', 'mag-x', 'mag-y', 'mag-z']].interpolate()
        cols = keep_cols(data, sensor_position=sensor_position)
        data[cols] = data[cols].interpolate()
        data[["user-id", 'label']] = data[["user-id", 'label']].ffill()

        # Convert time to datetime format
        data["time"] = pd.to_datetime(data["time"], unit='s')
        # print(f"Pamap2 sampling frequency is {1 / data['time'].diff().median().total_seconds()} Hz")

        if resampling:
            # resampled_dataset = resample_data(har_dataset[har_dataset["user-id"] == user], old_sampling_rate, new_sampling_rate)
            data = resample_data(data, old_sampling_rate, new_sampling_rate)
        """
        # Resample if ture
        if resampling is True:
            data = Downsampling(data, new_sampling_rate=13, old_sampling_rate=sampling_rate)
            data_base = pd.concat([data_base, data])
        else:
            data_base = pd.concat([data_base, data], ignore_index=True)
        """
        # Rename the activities
        data.loc[data['label'] == 0, "label"] = activity_list[0]
        data.loc[data['label'] == 1, "label"] = activity_list[1]
        data.loc[data['label'] == 2, "label"] = activity_list[2]
        data.loc[data['label'] == 3, "label"] = activity_list[3]
        data.loc[data['label'] == 4, "label"] = activity_list[4]
        data.loc[data['label'] == 5, "label"] = activity_list[5]
        data.loc[data['label'] == 6, "label"] = activity_list[6]
        data.loc[data['label'] == 7, "label"] = activity_list[7]
        data.loc[data['label'] == 12, "label"] = activity_list[8]
        data.loc[data['label'] == 13, "label"] = activity_list[9]
        data.loc[data['label'] == 16, "label"] = activity_list[10]
        data.loc[data['label'] == 17, "label"] = activity_list[11]
        data.loc[data['label'] == 24, "label"] = activity_list[12]

        # user_datasets[user] = get_user_datasets(data, user, axis=['acc-x', 'acc-y', 'acc-z'])
        user_datasets[user] = get_user_datasets(data, user, sensor_position=sensor_position)

    return user_datasets

def process_MHEALTH_files(
    data_folder_path,
    sensor_type='watch',
    old_sampling_rate=50,
    new_sampling_rate=100,
    resampling=False,
    use_chest_acc=False,
    sensor_position=None,
):
    """
    Process the MHEALTH dataset and return a dict of per-user DataFrames
    with standardized column names like:
        acc_x_forearm, acc_y_forearm, acc_z_forearm,
        gyr_x_forearm, ..., mag_z_forearm,
        acc_x_ankle, ..., mag_z_ankle (leg sensor)
    Optionally:
        acc_x_chest, acc_y_chest, acc_z_chest

    Args:
        data_folder_path (str): Path to the MHEALTH dataset folder (without .zip).
        old_sampling_rate (int): Original sampling rate of the dataset (Hz).
        new_sampling_rate (int): Target sampling rate (Hz) if resampling is True.
        resampling (bool): Whether to resample to `new_sampling_rate`.
        use_chest_acc (bool): If True, include chest accelerometer signals.

    Returns:
        dict[str, pd.DataFrame]: Mapping from user-id to processed DataFrame.
    """

    # Activities list
    activity_list = [
        'null',
        'Standing still',
        'Sitting and relaxing',
        'Lying down',
        'Walking',
        'Climbing stairs',
        'Waist bends forward',
        'Frontal elevation of arms',
        'Knees bending (crouching)',
        'Cycling',
        'Jogging',
        'Running',
        'Jump front & back'
    ]
    label_map = {i: name for i, name in enumerate(activity_list)}

    # Unzip the MHEALTH dataset
    zip_file_path = f"{data_folder_path}.zip"
    unzip_file(zip_file_path)

    # Adjust folder name to actual extracted directory
    data_folder_path = data_folder_path.replace('mhealth+dataset', 'MHEALTHDATASET')

    # Read dataset file names
    dataset_files = sorted(os.listdir(data_folder_path))

    # Get users_id (subject IDs)
    users_id = [
        f"subject{match.group(1)}"
        for match in (re.search(r'subject(\d+)', dataset_file) for dataset_file in dataset_files)
        if match
    ]

    user_datasets = {}

    for user in users_id:
        # Find the subject's log file
        matching = [s for s in dataset_files if f"{user}.log" in s]
        if not matching:
            continue
        matching = matching[0]

        # Original columns according to dataset description
        column_names = [
            'chest_acc_x', 'chest_acc_y', 'chest_acc_z',      # 1–3
            'ecg_lead1', 'ecg_lead2',                         # 4–5
            'left_ankle_acc_x', 'left_ankle_acc_y', 'left_ankle_acc_z',  # 6–8
            'left_ankle_gyro_x', 'left_ankle_gyro_y', 'left_ankle_gyro_z',  # 9–11
            'left_ankle_magnet_x', 'left_ankle_magnet_y', 'left_ankle_magnet_z',  # 12–14
            'right_lower_arm_acc_x', 'right_lower_arm_acc_y', 'right_lower_arm_acc_z',  # 15–17
            'right_lower_arm_gyro_x', 'right_lower_arm_gyro_y', 'right_lower_arm_gyro_z',  # 18–20
            'right_lower_arm_magnet_x', 'right_lower_arm_magnet_y', 'right_lower_arm_magnet_z',  # 21–23
            'label'  # 24
        ]

        data = pd.read_csv(
            os.path.join(data_folder_path, matching),
            delimiter='\t',
            header=None,
            names=column_names
        )

        # Add user-id
        data.insert(0, "user-id", user)

        # ===== Standardize column names =====
        # Map:
        #   left_ankle_*   -> *_ankle   (leg)
        #   right_lower_arm_* -> *_forearm (hand/arm)
        #   *_acc_*   -> acc_*
        #   *_gyro_*  -> gyr_*
        #   *_magnet_*-> mag_*
        rename_map = {
            # chest (optional)
            'chest_acc_x': 'acc_x_chest',
            'chest_acc_y': 'acc_y_chest',
            'chest_acc_z': 'acc_z_chest',

            # left ankle (leg)
            'left_ankle_acc_x': 'acc_x_ankle',
            'left_ankle_acc_y': 'acc_y_ankle',
            'left_ankle_acc_z': 'acc_z_ankle',
            'left_ankle_gyro_x': 'gyr_x_ankle',
            'left_ankle_gyro_y': 'gyr_y_ankle',
            'left_ankle_gyro_z': 'gyr_z_ankle',
            'left_ankle_magnet_x': 'mag_x_ankle',
            'left_ankle_magnet_y': 'mag_y_ankle',
            'left_ankle_magnet_z': 'mag_z_ankle',

            # right lower arm (forearm)
            'right_lower_arm_acc_x': 'acc_x_forearm',
            'right_lower_arm_acc_y': 'acc_y_forearm',
            'right_lower_arm_acc_z': 'acc_z_forearm',
            'right_lower_arm_gyro_x': 'gyr_x_forearm',
            'right_lower_arm_gyro_y': 'gyr_y_forearm',
            'right_lower_arm_gyro_z': 'gyr_z_forearm',
            'right_lower_arm_magnet_x': 'mag_x_forearm',
            'right_lower_arm_magnet_y': 'mag_y_forearm',
            'right_lower_arm_magnet_z': 'mag_z_forearm',
        }

        data = data.rename(columns=rename_map)

        # Select IMU channels in standardized naming
        sensor_cols = [
            # forearm (hand/arm)
            'acc_x_forearm', 'acc_y_forearm', 'acc_z_forearm',
            'gyr_x_forearm', 'gyr_y_forearm', 'gyr_z_forearm',
            'mag_x_forearm', 'mag_y_forearm', 'mag_z_forearm',
            # ankle (leg)
            'acc_x_ankle', 'acc_y_ankle', 'acc_z_ankle',
            'gyr_x_ankle', 'gyr_y_ankle', 'gyr_z_ankle',
            'mag_x_ankle', 'mag_y_ankle', 'mag_z_ankle',
        ]

        if use_chest_acc:
            sensor_cols = ['acc_x_chest', 'acc_y_chest', 'acc_z_chest'] + sensor_cols

        # Keep only relevant columns, and take a COPY to avoid SettingWithCopyWarning
        cols_to_keep = ['user-id', 'label'] + sensor_cols
        data = data.loc[:, cols_to_keep].copy()

        # Interpolate numeric sensor channels
        numeric_cols = [c for c in data.columns if c not in ['user-id', 'label']]
        data.loc[:, numeric_cols] = data.loc[:, numeric_cols].interpolate()

        # Forward-fill user-id and label
        data.loc[:, ['user-id', 'label']] = data.loc[:, ['user-id', 'label']].ffill()

        # Optional resampling
        if resampling:
            data = resample_data(data, old_sampling_rate, new_sampling_rate)

        # Map numeric labels to activity names (keep unmapped as-is)
        data['label'] = data['label'].map(label_map).fillna(data['label'])

        # Build per-user dataset (same style as Pamap2 / RealWorld)
        user_datasets[user] = get_user_datasets(data, user, sensor_position=sensor_position)

    return user_datasets


def process_MHEALTH_accelerometer_files(data_folder_path, sensor_type='watch', old_sampling_rate=50,
                                        new_sampling_rate=100, resampling=False, sensor_position=None,
                                        axis=['right_lower_arm_acc_x', 'right_lower_arm_acc_y', 'right_lower_arm_acc_z']):

    # Activities list
    activity_list = ['null', 'Standing still', 'Sitting and relaxing', 'Lying down', 'Walking', 'Climbing stairs',
                     'Waist bends forward', 'Frontal elevation of arms', 'Knees bending (crouching)', 'Cycling',
                     'Jogging', 'Running', 'Jump front & back']

    # Unzip the pamap2 dataset
    zip_file_path = f'{data_folder_path}.zip'
    unzip_file(zip_file_path)

    # Read datasets' names in dataset files
    data_folder_path = data_folder_path.replace('mhealth+dataset', 'MHEALTHDATASET')
    dataset_files = sorted(os.listdir(data_folder_path))


    # Get users_id
    users_id = users_id = [f"subject{match.group(1)}" for match in (re.search(r'subject(\d+)', dataset_file) for
                                                                    dataset_file in dataset_files) if match]

    data_base = pd.DataFrame()

    # Create the user datasets to save it as pkl file
    user_datasets = {}

    for user in users_id:

        # Read user data
        matching = [s for s in dataset_files if f'{user}.log' in s][0]

        column_names = [
            'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
            'ecg_lead1', 'ecg_lead2',
            'left_ankle_acc_x', 'left_ankle_acc_y', 'left_ankle_acc_z',
            'left_ankle_gyro_x', 'left_ankle_gyro_y', 'left_ankle_gyro_z',
            'left_ankle_magnet_x', 'left_ankle_magnet_y', 'left_ankle_magnet_z',
            'right_lower_arm_acc_x', 'right_lower_arm_acc_y', 'right_lower_arm_acc_z',
            'right_lower_arm_gyro_x', 'right_lower_arm_gyro_y', 'right_lower_arm_gyro_z',
            'right_lower_arm_magnet_x', 'right_lower_arm_magnet_y', 'right_lower_arm_magnet_z',
            'label'
        ]
        data = pd.read_csv(f"{data_folder_path}/{matching}", delimiter='\t', header=None, names=column_names)

        # Select the labels and the target axes
        data = data.loc[:, ['label'] + axis]

        # Rename the columns
        data.insert(0, "user-id", user)
        data = data.rename(columns={axis[0]: "x-axis", axis[1]: "y-axis", axis[2]: "z-axis"})
        data = data[['label', "x-axis", "y-axis", "z-axis", "user-id"]]

        # Interpolate the nan
        data[["x-axis", "y-axis", "z-axis"]] = data[["x-axis", "y-axis", "z-axis"]].interpolate()
        data[["user-id", 'label']] = data[["user-id", 'label']].ffill()


        if resampling:
            # resampled_dataset = resample_data(har_dataset[har_dataset["user-id"] == user], old_sampling_rate, new_sampling_rate)
            data = resample_data(data, old_sampling_rate, new_sampling_rate)

        # Rename the activities
        data.loc[data['label'] == 0, "label"] = activity_list[0]
        data.loc[data['label'] == 1, "label"] = activity_list[1]
        data.loc[data['label'] == 2, "label"] = activity_list[2]
        data.loc[data['label'] == 3, "label"] = activity_list[3]
        data.loc[data['label'] == 4, "label"] = activity_list[4]
        data.loc[data['label'] == 5, "label"] = activity_list[5]
        data.loc[data['label'] == 6, "label"] = activity_list[6]
        data.loc[data['label'] == 7, "label"] = activity_list[7]
        data.loc[data['label'] == 8, "label"] = activity_list[8]
        data.loc[data['label'] == 9, "label"] = activity_list[9]
        data.loc[data['label'] == 10, "label"] = activity_list[10]
        data.loc[data['label'] == 11, "label"] = activity_list[11]
        data.loc[data['label'] == 12, "label"] = activity_list[12]

        user_datasets[user] = get_user_datasets(data, user, sensor_position=sensor_position)

    return user_datasets

def synchronize_imus(imu_data_list, freq_hz=50):
    """
    Synchronizes multiple IMU datasets based on their Unix Time Millis timestamps.

    Args:
        imu_data_list (list): A list of pandas DataFrames, where each DataFrame
                              represents data from a single IMU. Each DataFrame
                              must have a 'timestamp_millis' column (Unix Time Millis)
                              and other IMU data columns (e.g., 'accel_x', 'gyro_y').
        freq_hz (int, optional): The desired output frequency in Hz. Defaults to 50 Hz.

    Returns:
        pandas.DataFrame: A single synchronized DataFrame containing all IMU data,
                          resampled to a common time base, without the timestamp column.
    """
    if not imu_data_list:
        return pd.DataFrame()

    # Convert timestamps to datetime and set as index for resampling
    processed_data = []
    for i, imu_df in enumerate(imu_data_list):
        imu_df['timestamp_seconds'] = imu_df['time'] / 1000
        imu_df['datetime'] = pd.to_datetime(imu_df['time'], unit='ms')
        imu_df = imu_df.set_index('datetime').sort_index()
        # Rename columns to avoid conflicts, e.g., accel_x_imu1, accel_x_imu2
        imu_df = imu_df.add_suffix(f'_imu{i+1}')
        processed_data.append(imu_df)

    # Determine a common time range and frequency for resampling
    min_time = max(df.index.min() for df in processed_data)
    max_time = min(df.index.max() for df in processed_data)

    # Convert freq_hz to a pandas frequency string (e.g., '10ms' for 100Hz)
    common_freq = f'{1000 / freq_hz:.0f}ms'

    # Create a common time index
    common_time_index = pd.date_range(start=min_time, end=max_time, freq=common_freq)

    # Resample and merge dataframes
    synchronized_df = pd.DataFrame(index=common_time_index)

    for df in processed_data:
        # Resample and interpolate. 'linear' is a common choice.
        # You might consider more advanced interpolation methods or filtering.
        resampled_df = df.reindex(common_time_index, method='nearest').interpolate(method='linear')
        synchronized_df = synchronized_df.merge(resampled_df, how='left', left_index=True, right_index=True)

    # Drop the original timestamp_millis and timestamp_seconds columns if they were suffixed
    # and are now redundant in the merged dataframe
    cols_to_drop = [col for col in synchronized_df.columns if 'timestamp_millis' in col or 'timestamp_seconds' in col
                    or 'time' in col]
    synchronized_df = synchronized_df.drop(columns=cols_to_drop, errors='ignore')
    synchronized_df = synchronized_df.rename(columns=lambda col: col.split('_imu', 1)[0])

    # The user wants only IMU data, so we don't reset the index to a timestamp column
    return synchronized_df

def linear_synchronization(streams, sampling_rate=50.0):
    """
    streams: dict[name -> DataFrame] each with 'time' column in ms and sensor channels
    sampling_rate: desired output rate in Hz (e.g. 50.0)
    returns: one DataFrame at uniform sampling, all channels renamed “col_name_stream”
    """
    # 1) Parse ms timestamps & index
    for name, df in streams.items():
        try:
            df['time'] = pd.to_datetime(df['time'], unit='ms')
        except:
            pass
        df.sort_values('time', inplace=True)
        df.set_index('time', inplace=True)

    # 2) Compute overlap window
    t0 = max(df.index.min() for df in streams.values())
    t1 = min(df.index.max() for df in streams.values())

    # 3) Build uniform time grid
    period_ms = int(1000.0 / sampling_rate)    # 20
    freq_str  = f"{period_ms}ms"                   # "20ms"
    grid      = pd.date_range(start=t0, end=t1, freq=freq_str)

    # 4) linear‐spline each stream onto the grid
    aligned = []
    # Pre‐compute target times in seconds for interp1d
    target_sec = grid.astype(np.int64) * 1e-9

    for name, df in streams.items():
        # restrict to overlap
        df_clip = df.loc[t0:t1]

        # original times in seconds
        orig_sec = df_clip.index.astype(np.int64) * 1e-9

        # build an empty DataFrame with the target grid
        df_linear = pd.DataFrame(index=grid)

        # fit & evaluate a spline for each channel
        for col in df_clip.columns:
            y = df_clip[col].values

            # drop any NaNs in the raw data
            valid = ~np.isnan(y)

            f_linear = interp1d(
                orig_sec[valid],
                y[valid],
                kind='linear',
                fill_value='extrapolate',
                assume_sorted=True
            )
            df_linear[col] = f_linear(target_sec)

        # rename columns to keep track of the stream
        aligned.append(df_linear)

    # 5) Combine all streams side by side
    combined = pd.concat(aligned, axis=1)
    combined = combined.dropna(axis=0, how='any')
    return combined

def process_RealWorld_HAR_files(data_folder_path, sensor_type='watch', old_sampling_rate=50, new_sampling_rate=100,
                                resampling=False, sensor_position=None):

    """
    Walks each proband's folder under `data_folder_path`, finds .zip files containing
    'acc', 'gyr', or 'mag' (excluding 'sqlite'), extracts the 'forearm' CSV inside,
    renames attr_x/y/z to acc-/gyr-/mag- axes, interpolates NaNs, tags with user-id
    and label, then **synchronizes** the three sensor streams by timestamp before
    optional resampling. Returns a dict of per-user DataFrames.
    """
    # 1) Locate each proband's data directory
    dataset_dirs = sorted([
        d for d, _, _ in os.walk(data_folder_path.replace('/realworld2016_dataset', ''))
        if d.endswith(os.path.join('data'))
    ])
    # 2) Derive user IDs in same order
    users_id = []
    for d in dataset_dirs:
        m = re.search(r'proband(\d+)', d)
        users_id.append(f'proband{m.group(1)}' if m else None)

    # 3) Accumulate per-user
    user_datasets = {}
    for data_dir, user in zip(dataset_dirs, users_id):
        if user is None:
            continue

        # find all IMU zips
        zips = glob.glob(os.path.join(data_dir, '*.zip'))
        imu_zips = [z for z in zips if any(k in os.path.basename(z).lower() for k in ('acc', 'gyr', 'mag'))
                    and 'sqlite' not in z.lower()]
        # collect dfs per label
        label_map = {}
        for zippath in imu_zips:
            fname = os.path.basename(zippath).lower()
            # sensor prefix
            if 'acc' in fname:
                prefix = 'acc-'
            elif 'gyr' in fname:
                prefix = 'gyr-'
            elif 'mag' in fname:
                prefix = 'mag-'
            else:
                continue
            # open and read forearm CSV
            with zipfile.ZipFile(zippath) as zf:
                files = [n for n in zf.namelist() if n.lower().endswith('.csv') and ('forearm' in n.lower() or
                                                                                     'shin' in n.lower())]
                if not files:
                    continue
                    df_dic = {}
                for path in files:
                # path = files[0]
                    try:
                        df = pd.read_csv(zf.open(path))
                        prefix = fr"{fname.split('/')[-1].split('_', 1)[0]}_{path.split('_', 2)[-1].split('.', 1)[0]}"
                    except Exception:
                        continue
                    # rename axes
                    df = df.rename(columns={
                        'attr_time': 'time',
                        'attr_x': fr"{fname.split('_', 1)[0]}_x_{path.split('_', 2)[-1].split('.', 1)[0]}",
                        'attr_y': fr"{fname.split('_', 1)[0]}_y_{path.split('_', 2)[-1].split('.', 1)[0]}",
                        'attr_z': fr"{fname.split('_', 1)[0]}_z_{path.split('_', 2)[-1].split('.', 1)[0]}"
                    })\
                    # interpolate
                    df.update(df.drop(columns=['id', 'time']).interpolate())

                    # df[[prefix + '_x', prefix + '_y', prefix + '_z']] = df[
                    #    [prefix + '_x', prefix + '_y', prefix + '_z']].interpolate()
                    # parse label from filename
                    # label = os.path.basename(zippath.replace(fr'{data_dir}/', ""))
                    # label = re.sub(rf'^{prefix[:-1]}[_\-]?', '', label, flags=re.IGNORECASE)
                    # label = re.sub(r'[_\-].*$', '', label)
                    label = fname.split('_', -1)[1]
                    # drop unwanted
                    if 'id' in df.columns:
                        df = df.drop(columns=['id'])
                    # timestamp
                    # df['time'] = pd.to_datetime(df['time'], unit='ms')
                    # sort by time
                    # df = df.sort_values('time').reset_index(drop=True)
                    # store
                    label_map.setdefault(label, {})[prefix] = df

        sensor_position_rank = {'forearm': 0, 'shin': 1}
        sensor_rank = {'acc': 0, 'gyr': 1, 'mag': 2}

        for lbl, subs in label_map.items():
            label_map[lbl] = dict(
                sorted(subs.items(),
                       key=lambda kv: (sensor_position_rank[kv[0].split('_')[-1]],
                                       sensor_rank[kv[0].split('_')[0]]))
            )

        # 4) For each label, synchronize acc, gyr, mag by timestamp via index-alignment
        data_base = pd.DataFrame()
        for label, streams in label_map.items():
            """
            # require at least one sensor
            dfs = []
            for sensor, stream in streams.items():
                df_s = stream.set_index('time')
                # rename index for clarity
                dfs.append(df_s)
            # if not dfs:
            #     continue
            # merge all on time index (union of timestamps)
            merged = pd.concat(dfs, axis=1).sort_index()
            # forward/backward fill small gaps and interpolate
            merged = merged.interpolate(method='time').ffill().bfill()
            # Drop any timestamps closer than your design period
            deltas = merged.index.diff()
            expected_dt = pd.Timedelta(seconds=1 / old_sampling_rate) - pd.Timedelta(milliseconds=5)
            merged = merged.loc[(deltas >= expected_dt) | (deltas.isna())]
            # reset index
            merged = merged.reset_index()
            """
            merged = linear_synchronization(streams, sampling_rate=old_sampling_rate)
            # merged = synchronize_imus(list(streams.values()), freq_hz=old_sampling_rate)
            # ensure user-id and label columns exist
            merged.insert(0, 'user-id', user)
            merged.insert(1, 'label', label)
            if label == 'climbingup' and user == "proband2":
                continue
            # optional resample to uniform timeline
            if resampling:
                merged = resample_data(merged, old_sampling_rate, new_sampling_rate)
            merged.columns = merged.columns.str.replace('_2_', '_', regex=False)
            data_base = pd.concat([data_base, merged], ignore_index=True)
            # data_base = data_base._append(merged, ignore_index=True)

        # user_datasets[user] = get_user_datasets(data_base, user, axis=['acc-x', 'acc-y', 'acc-z'])
        user_datasets[user] = get_user_datasets(data_base, user, sensor_position=sensor_position)
        row_mask = data_base.isna().any(axis=None)
        # Get the row indices
        # rows_with_nan = data_base.index[row_mask].tolist()
        if row_mask:
            print(row_mask)
    return user_datasets

def process_ActiVAtE_accelerometer_files(data_folder_path, sensor_type='watch', old_sampling_rate=13, new_sampling_rate=100, resampling=False):

    # Activities list
    activity_list = ["lying", "sitting", "walking", "running"]

    # Specify the folder path containing the CSV files
    folder_path = r"C:\Users\Mohammad\Ph.D\run\original_datasets\ActiVAtE"

    # Initialize an empty list to store DataFrames
    dfs = []

    # Loop through all files in the folder
    for file_name in os.listdir(data_folder_path):
        if file_name.endswith('.csv'):
            # Extract user-id from the file name
            user_id = file_name.split('_')[0]

            # Read the CSV file into a DataFrame
            df = pd.read_csv(os.path.join(data_folder_path, file_name))
            #df = df.rename(columns={"Unnamed: 0": "time", "0": "x-axis", "1": "y-axis", "2": "z-axis"})

            # Add a "user-id" column to the DataFrame
            df['user-id'] = user_id

            # Append the DataFrame to the list
            dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    final_df = pd.concat(dfs, ignore_index=True)

    # Display the final DataFrame
    #print(final_df)

    """
    file_path = glob.glob(os.path.join(data_folder_path, '*.csv'))
    df = pd.read_csv(file_path[0])

    df = df[['time', 'user', 'acc_x', 'acc_y', 'acc_z', 'label']]
    df = df.rename(columns={"acc_x": "x-axis", "acc_y": "y-axis", "acc_z": "z-axis", "user": "user-id"})
"""
    # Get users_id
    users_id = final_df['user-id'].unique()

    # Create the user datasets to save it as pkl file
    user_datasets = {}

    for user in users_id:

        data = final_df.loc[final_df["user-id"] == user].reset_index(drop=True).copy()

        if resampling:
            # resampled_dataset = resample_data(har_dataset[har_dataset["user-id"] == user], old_sampling_rate, new_sampling_rate)
            data = resample_data(data, old_sampling_rate, new_sampling_rate)

        user_datasets[user] = get_user_datasets(data, user)

    return user_datasets
