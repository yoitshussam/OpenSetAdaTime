from typing import Dict, Tuple, Optional
from self_har_utilities import prepare_dataset, load_unlabelled_dataset  # same helpers you use

def load_labelled_and_unlabelled(
    labelled_dataset_path: str,
    unlabelled_dataset_path: Optional[str],
    *,
    window_size: int = 400,
    max_unlabelled_windows: Optional[int] = 40000,
    leave_one_subject_out_cross_validation: bool = False,
    test_subject_number: int = 0,
    activity_mapping: Dict[str, str] = None,
    verbose: int = 1,
    labelled_repeat: int = 1,
    source_users: Optional[list] = None,
    target_users: Optional[list] = None,
    label_list: Optional[list] = None,
) -> Tuple[Dict, Tuple[int, ...], int]:
    """
    Loads:
      - prepared_datasets['labelled'] with train/val/test
      - prepared_datasets['unlabelled'] if unlabelled path is given

    Args:
      source_users: explicit list of user IDs for the source (train) split.
          When set, overrides leave_one_subject_out / test_subject_number.
      target_users: explicit list of user IDs for the target (test) split.

    Returns:
      prepared_datasets, input_shape, output_shape
    """
    if activity_mapping is None:
        activity_mapping = {}

    prepared_datasets: Dict = {}

    # --- user split ---
    def get_fixed_split_users(
        har_users,
        leave_one_subject_out_cross_validation=leave_one_subject_out_cross_validation,
        test_subject_number=test_subject_number,
    ):
        # If explicit user lists are provided, use them directly
        if source_users is not None and target_users is not None:
            return (source_users, target_users)
        if leave_one_subject_out_cross_validation:
            test_users = har_users[test_subject_number]
            train_users = [u for u in har_users if u != test_users]
        else:
            test_users = har_users[1::5]
            train_users = [u for u in har_users if all(u != t for t in test_users)]
        return (train_users, test_users)

    # --- labelled (source) ---
    prepared_datasets['labelled'] = prepare_dataset(
        labelled_dataset_path,
        window_size,
        get_fixed_split_users,
        validation_split_proportion=0.1,
        verbose=verbose,
        activity_mapping=activity_mapping,
        label_list=label_list,
    )
    input_shape  = prepared_datasets['labelled']['input_shape']
    output_shape = prepared_datasets['labelled']['output_shape']  # number of classes

    # --- unlabelled (target) ---
    if unlabelled_dataset_path:
        prepared_datasets = load_unlabelled_dataset(
            prepared_datasets,
            unlabelled_dataset_path,
            window_size,
            labelled_repeat=labelled_repeat,
            max_unlabelled_windows=max_unlabelled_windows,
            verbose=verbose,
            activity_mapping=activity_mapping,
            label_list=label_list,
        )

        # Remove extra variants the helper may add
        for k in ("labelled_x_repeat", "unlabelled_combined"):
            if k in prepared_datasets:
                del prepared_datasets[k]

    return prepared_datasets, input_shape, output_shape

# prepared_datasets, input_shape, output_shape = load_labelled_and_unlabelled(
#     labelled_dataset_path="processed/processed_datasets/RealWorld_processed.pkl",
#     unlabelled_dataset_path="processed/processed_datasets/Pamap2_processed.pkl",  # or None if source-only
#     window_size=400,
#     max_unlabelled_windows=40000,
#     leave_one_subject_out_cross_validation=False,
#     test_subject_number=0,
#     activity_mapping={
        
#         'sitting':'sitting',
#         'Sitting and relaxing':'sitting',

#         'standing':'standing',
#         'Standing still':'standing',

#         'lying':'lying',
#         'Lying down':'lying',

#         'running':'running',
#         'Running':'running',

#         'walking':'walking',
#         'Walking':'walking',
#     },
#     verbose=1,
#     labelled_repeat=1,
# )
