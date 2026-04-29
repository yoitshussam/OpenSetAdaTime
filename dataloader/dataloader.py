import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import Dataset
from torchvision import transforms
from load_data import load_labelled_and_unlabelled
import os
import numpy as np


class Load_Dataset(Dataset):
    def __init__(self, dataset, dataset_configs):
        super().__init__()
        self.num_channels = dataset_configs.input_channels
        self.return_index = getattr(dataset_configs, 'da_method', '') in [
            "DANCE", "UniOT", "UniJDOT", "UniJDOT_NoJoint", "UniJDOT_THR", "UniJDOT_THR_NOJOINT"
        ]

        # Data is always a tuple (x, y) from prepare_dataset
        x_data, y_data = dataset

        # Convert to tensors
        x_data = torch.from_numpy(np.array(x_data)).float()
        y_data = torch.from_numpy(np.array(y_data))

        # Fix dimensions to (N, C, L)
        if x_data.dim() == 2:
            x_data = x_data.unsqueeze(1)
        elif x_data.dim() == 3 and x_data.shape[1] != self.num_channels:
            x_data = x_data.permute(0, 2, 1)

        # Normalization
        if getattr(dataset_configs, 'normalize', False):
            data_mean = torch.mean(x_data, dim=(0, 2))
            data_std = torch.std(x_data, dim=(0, 2))
            self.transform = transforms.Normalize(mean=data_mean, std=data_std)
        else:
            self.transform = None

        self.x_data = x_data

        # Convert one-hot labels to class indices
        if y_data.dim() == 2 and y_data.shape[1] > 1:
            y_data = torch.argmax(y_data, dim=1)
        elif y_data.dim() == 2 and y_data.shape[1] == 1:
            y_data = y_data.squeeze(1)
        self.y_data = y_data.long()

        self.len = x_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index]
        if self.transform:
            x = self.transform(x.reshape(self.num_channels, -1, 1)).reshape(x.shape)
        if self.return_index:
            return x, self.y_data[index], index
        return x, self.y_data[index]

    def __len__(self):
        return self.len



def remove_private_class(dataset, private_class):
    """Remove samples belonging to private classes from dataset (x, y) tuple."""
    x_data, y_data = dataset
    y_data = np.array(y_data)
    # Labels are one-hot from prepare_dataset, convert to indices for filtering
    if y_data.ndim == 2 and y_data.shape[1] > 1:
        class_indices = np.argmax(y_data, axis=1)
    else:
        class_indices = y_data.ravel()
    mask = ~np.isin(class_indices, private_class)
    return (x_data[mask], y_data[mask])


def data_generator(data_path, dataset_configs, hparams, flag, activity_mapping=None,
                    source_users=None, target_users=None, label_list=None):
    """
    Load data from .pkl files using load_labelled_and_unlabelled function.

    Args:
        data_path: Path to the dataset directory
        dataset_configs: Dataset configuration object
        hparams: Hyperparameters dictionary
        flag: 'source' or 'target'
        activity_mapping: Dict mapping raw activity names to canonical names.
            Controls which classes are included — only mapped activities are kept.
        source_users: explicit user list for source split (cross-user mode)
        target_users: explicit user list for target split (cross-user mode)
    """
    if activity_mapping is None:
        activity_mapping = {}

    dataset_name = os.path.basename(data_path)
    base_dir = os.path.dirname(data_path)
    pkl_path = os.path.join(base_dir, f"{dataset_name}_processed.pkl")

    prepared_datasets, _, _ = load_labelled_and_unlabelled(
        labelled_dataset_path=pkl_path,
        unlabelled_dataset_path=pkl_path if flag == 'target' else None,
        window_size=dataset_configs.sequence_len,
        activity_mapping=activity_mapping,
        verbose=0,
        source_users=source_users,
        target_users=target_users,
        label_list=label_list,
    )

    dataset_files = []
    if flag == 'source':
        dataset_files.append(prepared_datasets['labelled']['train'])
        dataset_files.append(prepared_datasets['labelled']['test'])
        dataset_files.append(prepared_datasets['labelled']['val'])
    else:
        dataset_files.append(prepared_datasets['unlabelled']['train'])
        dataset_files.append(prepared_datasets['unlabelled']['train'])

    datasets = [Load_Dataset(df, dataset_configs) for df in dataset_files]

    # Dataloaders
    data_loaders = []
    for idx, dataset in enumerate(datasets):
        is_test = (idx == 1)  # index 1 is always the test set
        # Balanced source sampling if configured
        if getattr(dataset_configs, 'src_balanced', False) and flag == 'source' and not is_test:
            class_sample_count = np.array(
                [len(np.where(dataset.y_data.numpy() == t)[0]) for t in np.unique(dataset.y_data.numpy())])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in dataset.y_data.numpy()])
            samples_weight = torch.from_numpy(samples_weight).double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            data_loaders.append(torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=hparams["batch_size"],
                drop_last=True,
                sampler=sampler,
                num_workers=0
            ))
        else:
            data_loaders.append(torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=hparams["batch_size"],
                shuffle=not is_test,
                drop_last=not is_test,
                num_workers=0
            ))

    return data_loaders


def few_shot_data_generator(data_loader, dataset_configs, num_samples=5):
    x_data = data_loader.dataset.x_data
    y_data = data_loader.dataset.y_data

    NUM_SAMPLES_PER_CLASS = num_samples
    NUM_CLASSES = len(torch.unique(y_data))

    counts = [y_data.eq(i).sum().item() for i in range(NUM_CLASSES)]
    samples_count_dict = {i: min(counts[i], NUM_SAMPLES_PER_CLASS) for i in range(NUM_CLASSES)}

    samples_ids = {i: torch.where(y_data == i)[0] for i in range(NUM_CLASSES)}
    selected_ids = {i: torch.randperm(samples_ids[i].size(0))[:samples_count_dict[i]] for i in range(NUM_CLASSES)}

    selected_x = torch.cat([x_data[samples_ids[i][selected_ids[i]]] for i in range(NUM_CLASSES)], dim=0)
    selected_y = torch.cat([y_data[samples_ids[i][selected_ids[i]]] for i in range(NUM_CLASSES)], dim=0)

    few_shot_dataset = Load_Dataset((selected_x.numpy(), selected_y.numpy()), dataset_configs)

    few_shot_loader = torch.utils.data.DataLoader(dataset=few_shot_dataset, batch_size=len(few_shot_dataset),
                                                  shuffle=False, drop_last=False, num_workers=0)

    return few_shot_loader
