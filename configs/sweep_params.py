"""
Hyperparameter search spaces for Optuna sweeps.

Format:
    ('categorical', [v1, v2, ...])       -> trial.suggest_categorical
    ('float', low, high, step)           -> trial.suggest_float(..., step=step)
    ('float_log', low, high)             -> trial.suggest_float(..., log=True)
    ('int', low, high)                   -> trial.suggest_int

Scope: universal domain adaptation methods, plus PDA methods (SPADA, PDAAN)
which need a PDA-specific dataset config (no target-private classes) — see
RealWorld_male_female_PDA. OSDA-only (OVANet, OSBP) methods remain excluded.
"""
sweep_train_hparams = {
    'num_epochs':    ('categorical', [10]),
    'num_epochs_pr': ('categorical', [10]),
    'batch_size':    ('categorical', [32, 64]),
    'weight_decay':  ('categorical', [1e-4, 5e-4]),
}

sweep_alg_hparams = {
    'UDA': {
        'learning_rate':   ('categorical', [1e-2, 1e-3, 5e-4, 1e-4]),
        'src_cls_loss_wt': ('float', 0.05, 5.0, 0.05),
        'domain_loss_wt':  ('float', 0.05, 2.0, 0.05),
        'w0':              ('float', 0.05, 1.0, 0.05),
    },
    'DANCE': {
        'learning_rate': ('categorical', [1e-2, 1e-3, 5e-4, 1e-4]),
        'eta':           ('float', 0.05, 1.0, 0.05),
        'margin':        ('float', 0.05, 1.0, 0.05),
    },
    'UniOT': {
        'learning_rate': ('categorical', [1e-2, 1e-3, 1e-4]),
        'gamma':         ('float', 0.05, 1.0, 0.05),
        'mu':            ('float', 0.05, 1.0, 0.05),
        'lam':           ('float', 0.05, 5.0, 0.05),
        'temp':          ('categorical', [0.05, 0.1]),
        'K':             ('categorical', [5, 10, 15, 20]),
        # MQ_size must be >> batch_size (n_batch = MQ_size // batch_size).
        # Dropped 64 to avoid n_batch=1 pathologies when batch_size=64.
        'MQ_size':       ('categorical', [128, 1000, 2000]),
    },
    'PPOT': {
        'learning_rate': ('categorical', [1e-2, 1e-3, 1e-4]),
        # tau is the EMA momentum on src_prototype; outside [0.3, 0.7] the
        # prototype either freezes or tracks only the latest batch.
        'tau':           ('float', 0.3, 0.7, 0.05),
        # tau1 is the confidence cutoff for counting "known" target samples
        # when estimating alpha; paper uses 0.9.
        'tau1':          ('float', 0.5, 0.95, 0.05),
        # tau2 is compared against normalized class weights ≤ 1; paper uses 1.0.
        'tau2':          ('float', 0.5, 1.0, 0.05),
        'alpha':         ('categorical', [0.01, 0.001]),
        'beta':          ('categorical', [0.01, 0.001]),
        'reg':           ('categorical', [0.01, 0.1]),
        'ot':            ('float', 0.05, 5.0, 0.05),
        # Paper sets eta_2=0.01 (small) and eta_3=2 (large): the negative-
        # entropy term must dominate to drive uncertainty on target unknowns.
        # Capping n_entropy at 1.0 was the bug — UNK collapsed to 0.
        'p_entropy':     ('float', 0.005, 0.5, 0.005),
        'n_entropy':     ('float', 0.1, 3.0, 0.1),
        'neg':           ('categorical', [0.2, 0.25, 0.3, 0.4, 0.5]),
        # thresh = paper's xi (eval-time rejection cutoff); paper uses 0.75.
        'thresh':        ('float', 0.3, 0.9, 0.05),
    },
    'UniJDOT': {
        'learning_rate': ('categorical', [1e-2, 1e-3, 5e-4, 1e-4]),
        'lamb':          ('float', 0.05, 5.0, 0.05),
        'alpha':         ('float', 0.05, 5.0, 0.05),
        'src_weight':    ('float', 0.05, 5.0, 0.05),
        'K':             ('categorical', [5, 10, 15, 20]),
        'n_batch':       ('categorical', [5, 10, 15, 20, 30]),
        'trg_mem_size':  ('categorical', [32, 64, 128]),
        # joint_decision and threshold_method are held constant in hparams.py
        # (single-value categoricals waste Optuna trial dimensions).
    },
    'RAINCOAT': {
        'learning_rate':       ('categorical', [1e-2, 1e-3, 5e-4, 1e-4]),
        'sinkhorn_eps':        ('categorical', [1e-2, 1e-3, 1e-4]),
        'dip_p_threshold':     ('categorical', [0.01, 0.05, 0.1]),
        'num_epochs_correct':  ('categorical', [20, 50, 100]),
    },
    'TSFA': {
        'learning_rate': ('categorical', [1e-3, 5e-4, 1e-4, 5e-5]),
        'lambda_sde':    ('float', 0.1, 5.0, 0.1),
        'lambda_osam':   ('float', 0.1, 5.0, 0.1),
    },
    # OSBP (Saito et al., ECCV 2018) — K+1 head with adversarial GRL.
    # Paper uses SGD with lr=1e-3 and lambda_adv=1.0; we sweep around those.
    'OSBP': {
        'learning_rate': ('categorical', [5e-3, 1e-3, 5e-4, 1e-4]),
        'lambda_adv':    ('float', 0.1, 5.0, 0.1),
    },
    # OVANet (Saito & Saenko, ICCV 2021) — closed-set head + one-vs-all head.
    # No algo-specific weighting knob beyond LR; paper uses 5e-4.
    'OVANet': {
        'learning_rate': ('categorical', [1e-3, 5e-4, 1e-4, 5e-5]),
    },
    # SPADA (Liu et al., IEEE TII 2021) — two-discriminator partial DA.
    # Same loss-weight knobs as UDA (it shares the GRL/BCE adversarial form).
    'SPADA': {
        'learning_rate':   ('categorical', [1e-2, 1e-3, 5e-4, 1e-4]),
        'src_cls_loss_wt': ('float', 0.05, 5.0, 0.05),
        'domain_loss_wt':  ('float', 0.05, 2.0, 0.05),
    },
    # PDAAN (Zhou et al., MST 2022) — class-weighted CE + complement entropy
    # + domain-enhanced adversarial. Ranges follow paper sec. 4.4:
    #   - lr grid in figure 17 spans [1e-4 .. 5e-2]; reported optimum 5e-3
    #   - alpha, beta grid in figure 18: {0.5, 1.0, 1.5, 2.0}; optimum (1, 1)
    # xi (eq. 4 exponent) and rho_init (initial expansion proportion) are not
    # swept in the paper; reasonable small-cardinality grids around the
    # implementation defaults.
    'PDAAN': {
        'learning_rate': ('categorical', [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]),
        'alpha':         ('categorical', [0.5, 1.0, 1.5, 2.0]),
        'beta':          ('categorical', [0.5, 1.0, 1.5, 2.0]),
        'xi':            ('categorical', [0.5, 1.0, 2.0, 3.0]),
        'rho_init':      ('categorical', [0.5, 1.0, 1.5, 2.0]),
        'class_weights_momentum': ('categorical', [0.5, 0.7, 0.9, 0.95, 0.99]),
    },
}
