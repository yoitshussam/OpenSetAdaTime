def get_dataset_class(dataset_name, scenario=None):
    """Return the dataset config class for the given source dataset.

    RealWorld_male / RealWorld_female pick a config based on `scenario`:
        - 'PDA':  RealWorld_male_female_PDA  (source = 4 shared + 2
          source-private, target = 4 shared — extras live in source)
        - 'OSDA': RealWorld_male_female_OSDA (source = 4 shared,
          target = 4 shared + 2 target-private — extras live in target)
        - anything else (UniDA / CLOSED): RealWorld_male_female
          (4 shared / 2 src-private / 2 trg-private)

    Everything else falls back to the generic ALL config.

    `scenario` defaults to None and is supplied by the trainer based on
    the algorithm's SCENARIO attribute, so callers don't need to think
    about it.
    """
    if dataset_name in ("RealWorld_male", "RealWorld_female"):
        if scenario == "PDA":
            resolved = "RealWorld_male_female_PDA"
        elif scenario == "OSDA":
            resolved = "RealWorld_male_female_OSDA"
        else:
            resolved = "RealWorld_male_female"
    else:
        resolved = "ALL"

    if resolved not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[resolved]


class ALL():
    def __init__(self):
        super(ALL, self).__init__()
        # data parameters
        self.num_classes = 4
        self.class_names = ['sitting', 'lying', 'running', 'walking']
        self.sequence_len = 150

        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 18
        self.stride = 1
        self.dropout = 0.1
        self.cnn_blocks = [
            {'kernel_size': 5, 'maxpool': False, 'dropout': True},
            {'kernel_size': 5, 'maxpool': False, 'dropout': True},
            {'kernel_size': 5, 'maxpool': False, 'dropout': True},
        ]

        # features
        self.mid_channels = 64
        self.final_out_channels = 96
        self.features_len = 1

        # FNO (Fourier Neural Operator) backbone settings
        self.isFNO = False          # set True automatically when --backbone FNO
        self.fourier_modes = 64     # number of low-frequency Fourier modes

        # discriminator (for UDA/UAN)
        self.disc_hid_dim = 100

        # open-set DA
        self.da_method = ""  # set by trainer, used by dataloader for return_index
        self.src_balanced = False  # enable balanced source sampling

        # Activity mappings: control which classes each domain gets.
        # Only activities present in the mapping are kept.
        # To create an open-set scenario, include extra classes in target.
        # Example: source has 4 classes, target has all 5 → "walking" is unknown.
        self.source_activity_mapping = {
            'sitting': 'sitting',
            'Sitting and relaxing': 'sitting',
            # 'standing': 'standing',
            # 'Standing still': 'standing',
            'lying': 'lying',
            'Lying down': 'lying',
            'running': 'running',
            'Running': 'running',
            'walking': 'walking',
            'Walking': 'walking',
        }
        self.target_activity_mapping = {
            'sitting': 'sitting',
            'Sitting and relaxing': 'sitting',
            # 'standing': 'standing',
            # 'Standing still': 'standing',
            'lying': 'lying',
            'Lying down': 'lying',
            'running': 'running',
            'Running': 'running',
            'walking': 'walking',
            'Walking': 'walking',
            # 'cycling': 'cycling',
            # "rope jumping":"rope jumping",

        }


class RealWorld_male_female(ALL):
    """Config for the RealWorld male -> female UniDA sweep.

    Class split over the 8 RealWorld activities:
        shared (4):          sitting, lying, running, walking
        source-private (2):  climbingdown, jumping
        target-private (2):  standing, climbingup  (treated as unknown)

    Standing is deliberately placed on the target-private side: it is the
    static posture most similar to sitting, so rejecting it as unknown is
    the hard case that actually stresses UniDA methods.

    num_classes = 6 = source-known count (classifier output dim). Target
    samples from the 2 private classes get label indices >= 6 via
    shared_label_list and are masked to -1 (unknown) by the evaluator.
    """
    def __init__(self):
        super().__init__()

        self.num_classes = 6
        self.class_names = [
            'sitting', 'lying', 'running', 'walking',
            'climbingdown', 'jumping',  # source-private
        ]

        # Source: 4 shared + 2 source-private
        self.source_activity_mapping = {
            'sitting': 'sitting',
            'lying': 'lying',
            'running': 'running',
            'walking': 'walking',
            'climbingdown': 'climbingdown',
            'jumping': 'jumping',
        }

        # Target: 4 shared + 2 target-private (unknown)
        self.target_activity_mapping = {
            'sitting': 'sitting',
            'lying': 'lying',
            'running': 'running',
            'walking': 'walking',
            'standing': 'standing',
            'climbingup': 'climbingup',
        }


class RealWorld_male_female_PDA(ALL):
    """Config for the RealWorld male -> female *partial DA* sweep.

    Class split over the 8 RealWorld activities:
        shared (4):          sitting, lying, running, walking
        source-private (2):  climbingdown, jumping
        target-private (0):  none

    PDA assumption: target classes are a strict subset of source classes.
    The classifier has 6 outputs (the source-known set); evaluation only
    sees the 4 shared classes in target, so the methods (SPADA / PDAAN)
    must learn to suppress the 2 source-private classes via class-level
    weighting / domain weighting.
    """
    def __init__(self):
        super().__init__()

        self.num_classes = 6
        self.class_names = [
            'sitting', 'lying', 'running', 'walking',
            'climbingdown', 'jumping',  # source-private
        ]

        # Source: 4 shared + 2 source-private
        self.source_activity_mapping = {
            'sitting': 'sitting',
            'lying': 'lying',
            'running': 'running',
            'walking': 'walking',
            'climbingdown': 'climbingdown',
            'jumping': 'jumping',
        }

        # Target: 4 shared (no target-private classes — true PDA setting)
        self.target_activity_mapping = {
            'sitting': 'sitting',
            'lying': 'lying',
            'running': 'running',
            'walking': 'walking',
        }


class RealWorld_male_female_OSDA(ALL):
    """Config for the RealWorld male -> female *open-set DA* sweep.

    Mirror of the PDA config — extras live in target instead of source:
        shared (4):          sitting, lying, running, walking
        source-private (0):  none
        target-private (2):  standing, climbingup  (treated as unknown)

    OSDA assumption: source classes are a subset of target classes. The
    classifier has 4 outputs (the source-known set); samples from the 2
    target-private classes get label indices >= 4 via shared_label_list
    and are masked to -1 (unknown) by the evaluator. Standing is placed
    on the target-private side because it is the static posture most
    similar to sitting — the hard case for OSDA.
    """
    def __init__(self):
        super().__init__()

        self.num_classes = 4
        self.class_names = ['sitting', 'lying', 'running', 'walking']

        # Source: 4 shared only
        self.source_activity_mapping = {
            'sitting': 'sitting',
            'lying': 'lying',
            'running': 'running',
            'walking': 'walking',
        }

        # Target: 4 shared + 2 target-private (unknown)
        self.target_activity_mapping = {
            'sitting': 'sitting',
            'lying': 'lying',
            'running': 'running',
            'walking': 'walking',
            'standing': 'standing',
            'climbingup': 'climbingup',
        }
