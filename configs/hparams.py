## Hyperparameters for open-set domain adaptation algorithms.
## Backbone-specific (UniDABench style): pick CNN or FNO via get_hparams_class.

def get_hparams_class(dataset_name, backbone="CNN"):
    """Return the hparams class for the given backbone.

    `dataset_name` is currently ignored — all datasets share the same hparam
    set, mirroring the prior single-class layout. `backbone` selects between
    `ALL_CNN` and `ALL_FNO`.
    """
    cls_name = f"ALL_{backbone}"
    if cls_name not in globals():
        raise NotImplementedError(
            f"No hparams class for backbone {backbone!r} (expected {cls_name})")
    return globals()[cls_name]


class ALL_CNN():
    def __init__(self):
        super(ALL_CNN, self).__init__()
        # Only keys that should override per-method values belong here.
        # weight_decay was previously here at 1e-4 — that silently overrode the
        # swept per-method weight_decay when training non-sweep (main.py).
        # During a sweep this didn't matter because Optuna's sampled value
        # overrode train_params again; non-sweep training was the casualty.
        self.train_params = {
            'num_epochs': 30,
            'num_epochs_pr': 20,
        }
        self.alg_hparams = {
            "DANCE": {
                "batch_size": 64,
                "eta": 0.05,
                "learning_rate": 0.0001,
                "margin": 0.6000000000000001,
                "num_epochs": 10,
                "num_epochs_pr": 10,
                "weight_decay": 0.0001,
            },
            "OSBP": {
                "batch_size": 64,
                "lambda_adv": 4.7,
                "learning_rate": 0.001,
                "num_epochs": 10,
                "num_epochs_pr": 10,
                "weight_decay": 0.0005,
            },
            "OVANet": {
                "batch_size": 64,
                "learning_rate": 5e-05,
                "num_epochs": 10,
                "num_epochs_pr": 10,
                "weight_decay": 0.0001,
            },
            "PDAAN": {
                "alpha": 0.5,
                "batch_size": 32,
                "beta": 2.0,
                "class_weights_momentum": 0.9,
                "learning_rate": 0.0001,
                "num_epochs": 10,
                "num_epochs_pr": 10,
                "rho_init": 2.0,
                "weight_decay": 0.0005,
                "xi": 3.0,
            },
            "PPOT": {
                "alpha": 0.001,
                "batch_size": 64,
                "beta": 0.01,
                "learning_rate": 0.0001,
                "n_entropy": 1.7000000000000002,
                "neg": 0.4,
                "num_epochs": 10,
                "num_epochs_pr": 10,
                "ot": 0.05,
                "p_entropy": 0.295,
                "reg": 0.1,
                "tau": 0.65,
                "tau1": 0.6000000000000001,
                "tau2": 0.5,
                "thresh": 0.9,
                "weight_decay": 0.0001,
            },
            "RAINCOAT": {
                "batch_size": 32,
                "dip_p_threshold": 0.1,
                "learning_rate": 0.0001,
                "num_epochs": 10,
                "num_epochs_correct": 50,
                "num_epochs_pr": 10,
                "sinkhorn_eps": 0.01,
                "weight_decay": 0.0005,
            },
            "SPADA": {
                "batch_size": 32,
                "domain_loss_wt": 0.8,
                "learning_rate": 0.001,
                "num_epochs": 10,
                "num_epochs_pr": 10,
                "src_cls_loss_wt": 0.25,
                "weight_decay": 0.0005,
            },
            "TSFA": {
                "batch_size": 32,
                "lambda_osam": 0.30000000000000004,
                "lambda_sde": 3.0000000000000004,
                "learning_rate": 0.0005,
                "num_epochs": 10,
                "num_epochs_pr": 10,
                "weight_decay": 0.0005,
            },
            "UDA": {
                "batch_size": 64,
                "domain_loss_wt": 4.25,
                "learning_rate": 0.001,
                "num_epochs": 10,
                "num_epochs_pr": 10,
                "src_cls_loss_wt": 4.2,
                "w0": 0.8500000000000001,
                "weight_decay": 0.0005,
            },
            "UniJDOT": {
                "K": 5,
                "alpha": 2.9,
                "batch_size": 64,
                "joint_decision": True,
                "lamb": 3.05,
                "learning_rate": 0.0001,
                "n_batch": 10,
                "num_epochs": 10,
                "num_epochs_pr": 10,
                "src_weight": 1.55,
                "threshold_method": "threshold_otsu",
                "trg_mem_size": 128,
                "weight_decay": 0.0001,
            },
            "UniOT": {
                "K": 15,
                "MQ_size": 128,
                "batch_size": 64,
                "gamma": 0.45,
                "lam": 0.05,
                "learning_rate": 0.001,
                "mu": 0.7000000000000001,
                "num_epochs": 10,
                "num_epochs_pr": 10,
                "temp": 0.05,
                "weight_decay": 0.0005,
            },
        }


class ALL_FNO():
    """FNO-backbone hparams. Filled in from the FNO sweep (sweep_new)."""
    def __init__(self):
        super(ALL_FNO, self).__init__()
        # Only keys that should override per-method values belong here.
        # weight_decay was previously here at 1e-4 — that silently overrode the
        # swept per-method weight_decay when training non-sweep (main.py).
        # During a sweep this didn't matter because Optuna's sampled value
        # overrode train_params again; non-sweep training was the casualty.
        self.train_params = {
            'num_epochs': 20,
            'num_epochs_pr': 20,
        }
        self.alg_hparams = {
    "DANCE": {
        "batch_size": 64,
        "eta": 0.1,
        "learning_rate": 0.0001,
        "margin": 0.7,
        "weight_decay": 0.0001
    },
    "OSBP": {
        "batch_size": 64,
        "lambda_adv": 7.7,
        "learning_rate": 0.001,
        "weight_decay": 0.0005
    },
    "OVANet": {
        "batch_size": 64,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001
    },
    "PDAAN": {
        "alpha": 1.5,
        "batch_size": 64,
        "beta": 1.5,
        "class_weights_momentum": 0.7,
        "learning_rate": 0.0001,
        "rho_init": 2.0,
        "weight_decay": 0.0001,
        "xi": 3.0
    },
    "PPOT": {
        "alpha": 0.001,
        "batch_size": 64,
        "beta": 0.01,
        "learning_rate": 5e-05,
        "n_entropy": 2.1,
        "neg": 0.5,
        "ot": 1.5000000000000002,
        "p_entropy": 0.13,
        "reg": 0.1,
        "tau": 0.4,
        "tau1": 0.55,
        "tau2": 0.65,
        "thresh": 0.65,
        "weight_decay": 0.0001
    },
    "RAINCOAT": {
        "batch_size": 64,
        "dip_p_threshold": 0.05,
        "learning_rate": 0.001,
        "num_epochs_correct": 20,
        "sinkhorn_eps": 0.001,
        "weight_decay": 0.0001
    },
    "SPADA": {
        "batch_size": 64,
        "domain_loss_wt": 0.7000000000000001,
        "learning_rate": 0.0001,
        "src_cls_loss_wt": 2.55,
        "weight_decay": 0.0001
    },
    "TSFA": {
        "batch_size": 64,
        "lambda_osam": 3.3000000000000003,
        "lambda_sde": 0.2,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001
    },
    "UDA": {
        "batch_size": 64,
        "domain_loss_wt": 3.95,
        "learning_rate": 0.001,
        "src_cls_loss_wt": 3.35,
        "w0": 0.8500000000000001,
        "weight_decay": 0.0001
    },
    "UniJDOT": {
        "K": 20,
        "alpha": 5,
        "batch_size": 32,
        "joint_decision": True,
        "lamb": 5,
        "learning_rate": 0.001,
        "n_batch": 5,
        "src_weight": 0.45,
        "threshold_method": "threshold_yen",
        "trg_mem_size": 64,
        "weight_decay": 0.0001
    },
    "UniOT": {
        "K": 20,
        "MQ_size": 2000,
        "batch_size": 32,
        "gamma": 0.5,
        "lam": 0.35000000000000003,
        "learning_rate": 0.001,
        "mu": 0.7000000000000001,
        "temp": 0.05,
        "weight_decay": 0.0005
    }
}
