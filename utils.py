import torch
import torch.nn.functional as F
from torch import nn as nn

import random
import os
import sys
import logging
import numpy as np
import pandas as pd
from shutil import copy
from datetime import datetime

from sklearn.metrics import classification_report, accuracy_score

import hashlib
import ot


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def generate_seed(keyword: str) -> int:
    """Generates a reproducible but diverse integer seed from a keyword."""
    return int(hashlib.sha256(keyword.encode()).hexdigest(), 16) % (2**32)


def fix_randomness(run):
    SEED = generate_seed(f"openset_experiment_seed_{run}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _logger(logger_name, level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def starting_logs(data_type, da_method, exp_log_dir, src_id, tgt_id, run_id):
    log_dir = os.path.join(exp_log_dir, src_id + "_to_" + tgt_id + "_run_" + str(run_id))
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug(f'Method:  {da_method}')
    logger.debug("=" * 45)
    logger.debug(f'Source: {src_id} ---> Target: {tgt_id}')
    logger.debug(f'Run ID: {run_id}')
    logger.debug("=" * 45)
    return logger, log_dir


def save_checkpoint(home_path, algorithm, log_dir, last_model, best_model):
    save_dict = {
        "last": last_model,
        "best": best_model
    }
    save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
    torch.save(save_dict, save_path)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)


def _calc_metrics(pred_labels, true_labels, log_dir, home_path, target_names):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    r = classification_report(true_labels, pred_labels, target_names=target_names, digits=6, output_dict=True)

    df = pd.DataFrame(r)
    accuracy = accuracy_score(true_labels, pred_labels)
    df["accuracy"] = accuracy
    df = df * 100

    file_name = "classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    return accuracy * 100, r["macro avg"]["f1-score"] * 100


def copy_Files(destination):
    destination_dir = os.path.join(destination, "MODEL_BACKUP_FILES")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("utils.py", os.path.join(destination_dir, "utils.py"))
    copy("dataloader/dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models/models.py", os.path.join(destination_dir, f"models.py"))
    copy(f"models/loss.py", os.path.join(destination_dir, f"loss.py"))
    copy("algorithms/algorithms.py", os.path.join(destination_dir, "algorithms.py"))
    copy(f"configs/data_model_configs.py", os.path.join(destination_dir, f"data_model_configs.py"))
    copy(f"configs/hparams.py", os.path.join(destination_dir, f"hparams.py"))
    copy(f"configs/sweep_params.py", os.path.join(destination_dir, f"sweep_params.py"))


def calculate_risk(target_model, risk_dataloader, device):
    if type(risk_dataloader) == tuple:
        x_data = torch.cat((risk_dataloader[0].dataset.x_data, risk_dataloader[1].dataset.x_data), axis=0)
        y_data = torch.cat((risk_dataloader[0].dataset.y_data, risk_dataloader[1].dataset.y_data), axis=0)
    else:
        x_data = risk_dataloader.dataset.x_data
        y_data = risk_dataloader.dataset.y_data

    feat = target_model.feature_extractor(x_data.float().to(device))
    pred = target_model.classifier(feat)
    cls_loss = F.cross_entropy(pred, y_data.long().to(device))
    return cls_loss.item()


class DictAsObject:
    def __init__(self, d):
        self.__dict__ = d

    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(f"'DictAsObject' object has no attribute '{name}'")


##################################################
##########  OT UTILITIES  ########################
##################################################

def sinkhorn(out, epsilon, sinkhorn_iterations):
    """
    from https://github.com/facebookresearch/swav
    """
    Q = torch.exp(out / epsilon).t()
    B = Q.shape[1]
    K = Q.shape[0]

    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    return Q.t()


def ubot_CCD(sim, beta, fake_size=0, fill_size=0, mode='minibatch', stopThr=1e-4):
    M = -sim
    alpha = ot.unif(sim.size(0))

    Q_st = ot.unbalanced.sinkhorn_knopp_unbalanced(alpha, beta, M.detach().cpu().numpy(),
                                                    reg=0.01, reg_m=0.5, stopThr=stopThr)
    Q_st = torch.from_numpy(Q_st).float().to(sim.device)

    sum_pi = torch.sum(Q_st)
    Q_st_bar = Q_st / sum_pi

    if mode == 'minibatch':
        Q_anchor = Q_st_bar[fake_size + fill_size:, :]
    if mode == 'all':
        Q_anchor = Q_st_bar

    wt_i, pseudo_label = torch.max(Q_anchor, 1)
    ws_j = torch.sum(Q_st_bar, 0)

    uniformed_index = Q_st_bar.size(1)
    conf_label = torch.where(wt_i > 1 / Q_st_bar.size(0), pseudo_label, uniformed_index)
    high_conf_label = conf_label.clone()
    source_private_label = torch.nonzero(ws_j < 1 / Q_st_bar.size(1))
    for i in source_private_label:
        high_conf_label = torch.where(high_conf_label == i, uniformed_index, high_conf_label)
    high_conf_label_id = torch.nonzero(high_conf_label != uniformed_index).view(-1)

    new_beta = torch.sum(Q_st_bar, 0).cpu().numpy()

    return high_conf_label_id, high_conf_label, conf_label, new_beta


def ubot_CCD2(sim, beta, fake_size=0, fill_size=0, mode='minibatch', stopThr=1e-4):
    M = -sim
    alpha = ot.unif(sim.size(0))

    Q_st = ot.unbalanced.mm_unbalanced(alpha, beta, M.detach().cpu().numpy(), reg_m=0.5, stopThr=stopThr)
    Q_st = torch.from_numpy(Q_st).float().to(sim.device)

    sum_pi = torch.sum(Q_st)
    Q_st_bar = Q_st / sum_pi

    if mode == 'minibatch':
        Q_anchor = Q_st_bar[fake_size + fill_size:, :]
    if mode == 'all':
        Q_anchor = Q_st_bar

    wt_i, pseudo_label = torch.max(Q_anchor, 1)
    ws_j = torch.sum(Q_st_bar, 0)

    uniformed_index = Q_st_bar.size(1)
    conf_label = torch.where(wt_i > 1 / Q_st_bar.size(0), pseudo_label, uniformed_index)
    high_conf_label = conf_label.clone()
    source_private_label = torch.nonzero(ws_j < 1 / Q_st_bar.size(1))
    for i in source_private_label:
        high_conf_label = torch.where(high_conf_label == i, uniformed_index, high_conf_label)
    high_conf_label_id = torch.nonzero(high_conf_label != uniformed_index).view(-1)

    new_beta = torch.sum(Q_st_bar, 0).cpu().numpy()

    return high_conf_label_id, high_conf_label, conf_label, new_beta


def adaptive_filling(ubot_feature_t, source_prototype, gamma, beta, fill_size, stopThr=1e-4):
    device = ubot_feature_t.device
    sim = torch.matmul(ubot_feature_t, source_prototype.t())
    max_sim, _ = torch.max(sim, 1)
    pos_id = torch.nonzero(max_sim > gamma).reshape(-1)
    pos_rate = pos_id.size(0) / max_sim.size(0)
    pos_num = pos_id.size(0)
    neg_num = max_sim.size(0) - pos_num
    if pos_rate <= 0.5:
        fake_size = neg_num - pos_num
        if fake_size > 0:
            high_conf_label_id, _, __, ___ = ubot_CCD(sim, beta, fake_size=0, fill_size=fill_size,
                                                       mode='all', stopThr=stopThr)
            if high_conf_label_id.size(0) > 0:
                select_id = torch.randint(0, high_conf_label_id.size(0), (fake_size,)).to(device)
                fill_pos = sim[high_conf_label_id[select_id], :]
                newsim = torch.cat([fill_pos, sim], 0)
            else:
                fake_size = 0
                newsim = sim
        else:
            newsim = sim
    else:
        fake_size = pos_num - neg_num
        if fake_size > 0:
            farthest_sproto_id = torch.argmin(sim, 1)
            fake_private = 0.5 * ubot_feature_t + 0.5 * source_prototype.data[farthest_sproto_id, :]
            fake_private = F.normalize(fake_private)
            select_id = torch.randint(0, fake_private.size(0), (fake_size,)).to(device)
            fill_neg = fake_private[select_id, :]
            fake_sim = torch.matmul(fill_neg, source_prototype.t())
            newsim = torch.cat([fake_sim, sim], 0)
        else:
            newsim = sim

    return newsim, fake_size


def adaptive_filling2(ubot_feature_t, source_prototype, gamma, beta, fill_size, stopThr=1e-4):
    device = ubot_feature_t.device
    sim = torch.matmul(ubot_feature_t, source_prototype.t())
    max_sim, _ = torch.max(sim, 1)
    pos_id = torch.nonzero(max_sim > gamma).reshape(-1)
    pos_rate = pos_id.size(0) / max_sim.size(0)
    pos_num = pos_id.size(0)
    neg_num = max_sim.size(0) - pos_num
    if pos_rate <= 0.5:
        fake_size = neg_num - pos_num
        if fake_size > 0:
            high_conf_label_id, _, __, ___ = ubot_CCD2(sim, beta, fake_size=0, fill_size=fill_size,
                                                        mode='all', stopThr=stopThr)
            if high_conf_label_id.size(0) > 0:
                select_id = torch.randint(0, high_conf_label_id.size(0), (fake_size,)).to(device)
                fill_pos = sim[high_conf_label_id[select_id], :]
                newsim = torch.cat([fill_pos, sim], 0)
            else:
                fake_size = 0
                newsim = sim
        else:
            newsim = sim
    else:
        fake_size = pos_num - neg_num
        if fake_size > 0:
            farthest_sproto_id = torch.argmin(sim, 1)
            fake_private = 0.5 * ubot_feature_t + 0.5 * source_prototype.data[farthest_sproto_id, :]
            fake_private = F.normalize(fake_private)
            select_id = torch.randint(0, fake_private.size(0), (fake_size,)).to(device)
            fill_neg = fake_private[select_id, :]
            fake_sim = torch.matmul(fill_neg, source_prototype.t())
            newsim = torch.cat([fake_sim, sim], 0)
        else:
            newsim = sim

    return newsim, fake_size


def run_kmeans(L2_feat, ncentroids, init_centroids=None, seed=None, gpu=False, min_points_per_centroid=1):
    import faiss
    if seed is None:
        seed = int(os.environ.get('PYTHONHASHSEED', '0'))
    dim = L2_feat.shape[1]
    kmeans = faiss.Kmeans(d=dim, k=ncentroids, seed=seed, gpu=gpu, niter=20, verbose=False,
                          nredo=5, min_points_per_centroid=min_points_per_centroid, spherical=True)
    if torch.is_tensor(L2_feat):
        L2_feat = variable_to_numpy(L2_feat)
    kmeans.train(L2_feat, init_centroids=init_centroids)
    _, pred_centroid = kmeans.index.search(L2_feat, 1)
    pred_centroid = np.squeeze(pred_centroid)
    return pred_centroid, kmeans.centroids


##################################################
##########  EASYDL INLINED UTILITIES  ############
##################################################

def variable_to_numpy(x):
    """Convert a torch tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


class TrainingModeManager:
    """Context manager to temporarily set modules to train/eval mode."""
    def __init__(self, modules, train=False):
        self.modules = modules if isinstance(modules, (list, tuple)) else [modules]
        self.train = train
        self.original_modes = []

    def __enter__(self):
        self.original_modes = [m.training for m in self.modules]
        for m in self.modules:
            m.train(self.train)
        return self

    def __exit__(self, *args):
        for m, mode in zip(self.modules, self.original_modes):
            m.train(mode)


class Accumulator:
    """Accumulate numpy arrays across batches."""
    def __init__(self):
        self.data = {}

    def updateData(self, d):
        for key, value in d.items():
            if key not in self.data:
                self.data[key] = []
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            self.data[key].append(value)

    def getByKey(self, key):
        return np.concatenate(self.data[key], axis=0)
