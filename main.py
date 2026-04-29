# =============================================================================
# Available preprocessed datasets (from SELF_HAR/run_datasets.py).
# Each dataset has a per-dataset processing function in raw_data_processing.py
# that turns raw sensor files into a {name}_processed.pkl consumed by the
# OpenSet-AdaTime dataloader. Below: dataset name, sampling rate, sensor, and
# the activity classes (label_list) it provides — use these names when
# building source_activity_mapping / target_activity_mapping in
# configs/data_model_configs.py.
#
# RealWorld HAR  (50 Hz, watch)
#   walking, running, sitting, standing, lying,
#   climbingup, climbingdown, jumping
#
# Pamap2  (100 Hz, watch; has null class)
#   lying, sitting, standing, walking, running, cycling, Nordic walking,
#   watching TV, computer work, car driving, ascending stairs,
#   descending stairs, vacuum cleaning, ironing, folding laundry,
#   house cleaning, playing soccer, rope jumping,
#   other (transient activities)
#
# MHEALTH  (50 Hz, watch; has null class — labels are numeric '0'..'12')
#   0=null, 1=Standing still, 2=Sitting and relaxing, 3=Lying down,
#   4=Walking, 5=Climbing stairs, 6=Waist bends forward,
#   7=Frontal elevation of arms, 8=Knees bending (crouching),
#   9=Cycling, 10=Jogging, 11=Running, 12=Jump front & back
#
# motionsense  (50 Hz, phone)
#   sit, std, wlk, ups, dws, jog
#   (sitting, standing, walking, walking upstairs, walking downstairs, jogging)
#
# Watch_hhar  (200 Hz, watch)
#   sit, stand, walk, stairsup, stairsdown, bike
#
# Phone_hhar  (100 Hz, phone)
#   sit, stand, walk, stairsup, stairsdown, bike
#
# (Wisdm and ActiVAtE are defined in run_datasets.py but currently commented
#  out / not in DATASET_METADATA, so they have no preprocessed pkl available.)
# =============================================================================

from trainers.train import Trainer
import mlflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")
from load_data import load_labelled_and_unlabelled
import argparse
parser = argparse.ArgumentParser()

if __name__ == "__main__":

    # ========  Experiments Name ================
    parser.add_argument('--save_dir',            default='experiments_logs', type=str, help='Directory containing all experiments')
    parser.add_argument('--exp_name',            default='EXP1',         type=str, help='experiment name')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method',           default='ALL',          type=str,
                        help='UDA, OVANet, DANCE, PPOT, UniOT, UniJDOT, RAINCOAT, OSBP, TSFA, SPADA, PDAAN, ALL')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path',           default=r'../dataset', type=str, help='Path containing datasets')
    parser.add_argument('--source_dataset',      default='RealWorld',    type=str, help='Dataset of choice')
    parser.add_argument('--target_dataset',      default='Pamap2',       type=str, help='Dataset of choice')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone',            default='FNO',          type=str, help='Backbone: CNN, FNO (CNN+Fourier)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs',            default=1,              type=int, help='Number of consecutive runs with different seeds')
    parser.add_argument('--device',              default="cuda",         type=str, help='cpu or cuda')

    # ========= Scenario selection ================
    parser.add_argument('--scenario',            default=None,           type=str,
                        help='Run all methods for a given scenario: OSDA, UniDA, PDA, CLOSED. '
                             'Overrides --da_method when set.')

    # Methods grouped by scenario (mirrors the SCENARIO class attribute on each Algorithm)
    SCENARIO_METHODS = {
        "OSDA":   ["OSBP", "TSFA"],
        "UniDA":  ["UDA", "OVANet", "DANCE", "PPOT", "UniOT", "UniJDOT", "RAINCOAT"],
        "PDA":    ["SPADA", "PDAAN"],
    }
    ALL_METHODS = [m for ms in SCENARIO_METHODS.values() for m in ms]

    args = parser.parse_args()

    # Resolve which methods to run + auto-set uniDA flag
    if args.scenario is not None:
        if args.scenario not in SCENARIO_METHODS:
            raise ValueError(f"Unknown scenario {args.scenario}. Choose from {list(SCENARIO_METHODS)}")
        run_methods = SCENARIO_METHODS[args.scenario]
    elif args.da_method == 'ALL':
        run_methods = ALL_METHODS
    else:
        run_methods = [args.da_method]

    for method in run_methods:
        args.da_method = method
        trainer = Trainer(args)
        trainer.fit()
