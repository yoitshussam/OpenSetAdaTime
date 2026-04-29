## Hyperparameters for open-set domain adaptation algorithms.
## Starting from EEG_CNN hparams in UniDABench, suitable for HAR time series.

def get_hparams_class(dataset_name):
    """Return the dataset class with the given name."""
    dataset_name = 'ALL'
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class ALL():
    def __init__(self):
        super(ALL, self).__init__()
        self.train_params = {
            'num_epochs': 30,
            'num_epochs_pr': 20,
            'weight_decay': 1e-4,
        }
        self.alg_hparams = {
    "DANCE": {
        "batch_size": 64,
        "eta": 0.05,
        "learning_rate": 0.0001,
        "margin": 0.6000000000000001,
        "num_epochs": 10,
        "num_epochs_pr": 10,
        "weight_decay": 0.0001
    },
    "OSBP": {
        "batch_size": 64,
        "lambda_adv": 4.7,
        "learning_rate": 0.001,
        "num_epochs": 10,
        "num_epochs_pr": 10,
        "weight_decay": 0.0005
    },
    "OVANet": {
        "batch_size": 64,
        "learning_rate": 5e-05,
        "num_epochs": 10,
        "num_epochs_pr": 10,
        "weight_decay": 0.0001
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
        "xi": 3.0
    },
    "PPOT": {
        "alpha": 0.001,
        "batch_size": 64,
        "beta": 0.001,
        "learning_rate": 0.0001,
        "n_entropy": 2.0,
        "neg": 0.25,
        "num_epochs": 10,
        "num_epochs_pr": 10,
        "ot": 5.0,
        "p_entropy": 0.01,
        "reg": 0.01,
        "tau": 0.65,
        "tau1": 0.9,
        "tau2": 1.0,
        "thresh": 0.75,
        "weight_decay": 0.0001
    },
    "RAINCOAT": {
        "batch_size": 32,
        "dip_p_threshold": 0.1,
        "learning_rate": 0.0001,
        "num_epochs": 10,
        "num_epochs_correct": 50,
        "num_epochs_pr": 10,
        "sinkhorn_eps": 0.01,
        "weight_decay": 0.0005
    },
    "SPADA": {
        "batch_size": 32,
        "domain_loss_wt": 0.8,
        "learning_rate": 0.001,
        "num_epochs": 10,
        "num_epochs_pr": 10,
        "src_cls_loss_wt": 0.25,
        "weight_decay": 0.0005
    },
    "TSFA": {
        "batch_size": 32,
        "lambda_osam": 0.30000000000000004,
        "lambda_sde": 3.0000000000000004,
        "learning_rate": 0.0005,
        "num_epochs": 10,
        "num_epochs_pr": 10,
        "weight_decay": 0.0005
    },
    "UDA": {
        "batch_size": 64,
        "domain_loss_wt": 4.25,
        "learning_rate": 0.001,
        "num_epochs": 10,
        "num_epochs_pr": 10,
        "src_cls_loss_wt": 4.2,
        "w0": 0.8500000000000001,
        "weight_decay": 0.0005
    },
    "UniJDOT": {
        "K": 5,
        "alpha": 2.9,
        "batch_size": 64,
        "lamb": 3.05,
        "learning_rate": 0.0001,
        "n_batch": 10,
        "num_epochs": 10,
        "num_epochs_pr": 10,
        "src_weight": 1.55,
        "trg_mem_size": 128,
        "weight_decay": 0.0001
    },
    "UniOT": {
        "K": 20,
        "MQ_size": 128,
        "batch_size": 64,
        "gamma": 0.4,
        "lam": 3.65,
        "learning_rate": 0.0001,
        "mu": 0.2,
        "num_epochs": 10,
        "num_epochs_pr": 10,
        "temp": 0.05,
        "weight_decay": 0.0001
    }
}