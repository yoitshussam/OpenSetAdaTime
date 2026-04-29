from trainers.sweep import Trainer
import mlflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")
import argparse
parser = argparse.ArgumentParser()

if __name__ == "__main__":

    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='ALL', type=str,
                        help='UDA, DANCE, PPOT, UniOT, UniJDOT, RAINCOAT, TSFA, '
                             'OSBP, OVANet, SPADA, PDAAN, ALL')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'../dataset', type=str, help='Path containing dataset')
    parser.add_argument('--source_dataset', default='RealWorld_male', type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')
    parser.add_argument('--target_dataset', default='RealWorld_female', type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone: CNN, TCN, RESNET18')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=3, type=int, help='Number of consecutive runs with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')
    parser.add_argument('--exp_name', default='sweep_complete', type=str, help='experiment name')

    # ======== Sweep settings =====================
    parser.add_argument('--num_sweeps', default=15, type=int, help='Number of Optuna trials')
    parser.add_argument('--hp_search_strategy', default="bayes", type=str,
                        help='Optuna sampler: random, grid, bayes (TPE)')
    parser.add_argument('--metric_to_minimize', default="H_score", type=str,
                        help='Metric to minimize: src_risk, trg_risk, H_score, '
                             'f1_score, acc. H_score is auto-substituted with '
                             'f1_score when --scenario is PDA/CLOSED.')

    # ========= Open-set DA settings ==============
    parser.add_argument('--uniDA', default=True, type=bool, help='Enable universal/open-set DA evaluation')

    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='experiments_logs/sweep_logs', type=str,
                        help='Directory containing all experiments')

    # ========= Scenario selection ================
    # When set, runs every method registered under that scenario and ignores
    # --da_method. Each method's SCENARIO attribute then drives the dataset
    # config (e.g. PDA -> RealWorld_male_female_PDA).
    parser.add_argument('--scenario', default=None, type=str,
                        help='Run all sweep methods for a scenario: OSDA, UniDA, PDA. '
                             'Overrides --da_method when set.')

    # Methods grouped by scenario (mirrors the SCENARIO class attribute on each Algorithm).
    SCENARIO_METHODS = {
        "OSDA":   ["OSBP", "TSFA"],
        "UniDA":  ["UDA", "OVANet", "DANCE", "PPOT", "UniOT", "UniJDOT", "RAINCOAT"],
        "PDA":    ["SPADA", "PDAAN"],
    }
    ALL_METHODS = [m for ms in SCENARIO_METHODS.values() for m in ms]

    args = parser.parse_args()

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
        trainer.sweep()
