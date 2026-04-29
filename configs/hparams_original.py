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
            'NO_ADAPT': {
                'batch_size': 32,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'step_size': 50,
                'lr_decay': 0.5,
                'src_cls_loss_wt': 1,
            },
            'TARGET_ONLY': {
                'batch_size': 32,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'step_size': 50,
                'lr_decay': 0.5,
                'src_cls_loss_wt': 1,
            },
            'UDA': {
                'batch_size': 64,
                'domain_loss_wt': 1,
                'learning_rate': 1e-4,
                'num_epochs_pr': 20,
                'src_cls_loss_wt': 3.95,
                'w0': 0.95,
                'weight_decay': 1e-4,
            },
            'OVANet': {
                'batch_size': 64,
                'learning_rate': 5e-4,
                'num_epochs_pr': 20,
                'weight_decay': 1e-4,
            },
            'DANCE': {
                'batch_size': 64,
                'eta': 0.05,
                'learning_rate': 1e-4,
                'margin': 0.55,
                'num_epochs_pr': 20,
                'weight_decay': 1e-4,
            },
            'PPOT': {
                'alpha': 0.01,
                'batch_size': 32,
                'beta': 0.001,
                'learning_rate': 0.01,
                'n_entropy': 0.1,
                'neg': 0.4,
                'num_epochs_pr': 20,
                'ot': 0.4,
                'p_entropy': 0.6,
                'reg': 0.1,
                'tau': 0.35,
                'tau1': 0.6,
                'tau2': 1.6,
                'thresh': 1,
                'weight_decay': 1e-4,
            },
            'UniOT': {
                'K': 15,
                'MQ_size': 1000,
                'batch_size': 32,
                'gamma': 0.9,
                'lam': 1.6,
                'learning_rate': 1e-4,
                'mu': 1,
                'num_epochs_pr': 20,
                'temp': 0.05,
                'weight_decay': 1e-4,
            },
            'UniJDOT': {
                'K': 5,
                'alpha': 4.05,
                'batch_size': 64,
                'joint_decision': True,
                'lamb': 1.65,
                'learning_rate': 5e-4,
                'n_batch': 10,
                'num_epochs_pr': 20,
                'src_weight': 3.8,
                'threshold_method': 'threshold_yen',
                'trg_mem_size': 32,
                'weight_decay': 1e-4,
            },
            'RAINCOAT': {
                'batch_size': 32,
                'learning_rate': 5e-4,
                'weight_decay': 1e-4,
                'num_epochs_correct': 50,
                'kernel_size': 5,
                'fourier_modes': 75,
                'sinkhorn_eps': 1e-3,
                'dip_p_threshold': 0.05,
            },
            'OSBP': {
                'batch_size': 64,
                'learning_rate': 1e-3,
                'weight_decay': 5e-4,
                'num_epochs_pr': 20,
                'lambda_adv': 1.0,
            },
            'TSFA': {
                'batch_size': 32,
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
                'num_epochs_pr': 20,
                'sde_hidden_dim': 128,
                'sde_num_steps': 10,
                'lambda_sde': 1.0,
                'lambda_osam': 1.0,
            },
            'SPADA': {
                'batch_size': 64,
                'learning_rate': 1e-3,
                'weight_decay': 5e-4,
                'src_cls_loss_wt': 1.0,
                'domain_loss_wt': 1.0,
            },
            'PDAAN': {
                'batch_size': 64,
                'learning_rate': 5e-3,
                'weight_decay': 1e-4,
                'alpha': 1.0,
                'beta': 1.0,
                'xi': 1.0,
                'rho_init': 1.0,
                'class_weights_momentum': 0.9,
            },
        }
