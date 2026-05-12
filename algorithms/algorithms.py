import copy
import ot
import torch
import torch.nn as nn
import numpy as np
import itertools
import skimage.filters as sfil

from sklearn.cluster import KMeans
from torch.nn import BCELoss
from tqdm import tqdm

from models.models import classifier, ReverseLayerF, \
    DiscriminatorUDA, classifierOVANet, classifierOSBP, LinearAverage, CLS, \
    ProtoCLS, MemoryQueue, ClassMemoryQueue
from models.raincoat_modules import TFEncoder, TFDecoder, RaincoatClassifier
from models.tsfa_modules import TFFeatureExtractor, NeuralSDE
from models.loss import ConditionalEntropyLoss, Entropy, SinkhornDistance
from utils import adaptive_filling, ubot_CCD, sinkhorn, ubot_CCD2, adaptive_filling2, \
    run_kmeans, variable_to_numpy, TrainingModeManager, Accumulator
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import torch.nn.functional as F

try:
    from diptest import diptest as dip_test
    HAS_DIPTEST = True
except ImportError:
    HAS_DIPTEST = False


class classifierNoBias(nn.Module):
    """Simple linear classifier without bias, used by DANCE."""
    def __init__(self, configs):
        super(classifierNoBias, self).__init__()
        self.fc = nn.Linear(configs.feat_dim, configs.num_classes, bias=False)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.

    Each subclass declares its scenario via the class attribute
        SCENARIO in {"CLOSED", "PDA", "OSDA", "UniDA"}
    `uniDA` is derived from SCENARIO — single source of truth. OSDA/UniDA
    scenarios get unknown-class handling in eval; CLOSED/PDA scenarios use
    plain closed-set CE.
    """
    SCENARIO = "UniDA"  # default; subclasses override

    @property
    def uniDA(self):
        return self.SCENARIO in ("OSDA", "UniDA")

    def __init__(self, configs, backbone):
        super(Algorithm, self).__init__()
        self.configs = configs

        self.cross_entropy = nn.CrossEntropyLoss()
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()),
        )

    # update function is common to all algorithms
    def update(self, src_loader, trg_loader, avg_meter, val_loader, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            losses = {'Pr_Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

    def get_latent_features(self, dataloader):
        feature_set = []
        pred_set = []
        label_set = []
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (data, label) in enumerate(dataloader):
                data = data.to(self.device)
                feature = self.feature_extractor(data)
                pred = F.softmax(self.classifier(feature))
                pred_set.append(pred.cpu())
                feature_set.append(feature.cpu())
                label_set.append(label.cpu())
            feature_set = torch.cat(feature_set, dim=0)
            pred_set = torch.cat(pred_set, dim=0)
            feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
        return feature_set, label_set, pred_set

    def evaluate(self, test_loader, trg_private_class, src=False):
        feature_extractor = self.feature_extractor.to(self.device)
        classifier = self.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = feature_extractor(data)
                predictions = classifier(features)

                # compute loss
                if self.uniDA:
                    if src:
                        corr_preds = self.correct_predictions(predictions)
                        loss = F.cross_entropy(corr_preds, labels)
                    else:
                        m = torch.isin(labels.cpu(), trg_private_class, invert=True)
                        loss = F.cross_entropy(predictions[m], labels[m])
                else:
                    loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.detach().cpu().item())
                pred = predictions.detach()

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)
        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        return loss, full_preds, full_labels

    def correct_predictions(self, preds):
        return preds

    def decision_function(self, preds):
        confidence, pred = preds.max(dim=1)
        return pred

    # train loop vary from one method to another
    def training_epoch(self, *args, **kwargs):
        raise NotImplementedError


class NO_ADAPT(Algorithm):
    """
    Lower bound: train on source and test on target.
    """
    SCENARIO = "CLOSED"
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        for src_x, src_y in src_loader:

            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))


class TARGET_ONLY(Algorithm):
    """
    Upper bound: train on target and test on target.
    """
    SCENARIO = "CLOSED"

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        for trg_x, trg_y in trg_loader:

            trg_x, trg_y = trg_x.to(self.device), trg_y.to(self.device)

            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            trg_cls_loss = self.cross_entropy(trg_pred, trg_y)

            loss = trg_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Trg_cls_loss': trg_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, trg_x.size(0))

        self.lr_scheduler.step()


class UDA(Algorithm):
    SCENARIO = "UniDA"

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Domain Discriminator
        self.domain_classifier = DiscriminatorUDA(configs)
        self.adv_discriminator = DiscriminatorUDA(configs)

        self.conditional_entropy = ConditionalEntropyLoss()
        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.adv_discriminator.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.w_0 = hparams["w0"]

        self.bce = BCELoss()

    def normalize_weight(self, x):
        min_val = x.min()
        max_val = x.max()
        x = (x - min_val) / (max_val - min_val)
        x = x / torch.mean(x)
        return x.detach()

    def reverse_sigmoid(self, y):
        return torch.log(y / (1.0 - y + 1e-10) + 1e-10)

    def get_src_weights(self, domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
        before_softmax = before_softmax / class_temperature
        after_softmax = nn.Softmax(-1)(before_softmax)
        domain_logit = self.reverse_sigmoid(domain_out)
        domain_logit = domain_logit / domain_temperature
        domain_out = nn.Sigmoid()(domain_logit)

        entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
        entropy_norm = entropy / np.log(after_softmax.size(1))
        weight = entropy_norm - domain_out
        weight = weight.detach()
        return weight

    def get_trg_weights(self, domain_out, before_softmax, domain_temperature=1.0, class_temperature=1.0):
        return -1 * self.get_src_weights(domain_out, before_softmax, domain_temperature, class_temperature)

    def update(self, src_loader, trg_loader, avg_meter, val_loader, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs + 1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        with torch.no_grad():
            self.feature_extractor.eval()
            self.classifier.eval()
            X = src_loader.dataset.x_data.to(self.device)
            Y = src_loader.dataset.y_data.numpy()
            logits = self.classifier(self.feature_extractor(X))
            preds = logits.detach().cpu().argmax(axis=1).numpy()
        self.feature_extractor.train()
        self.classifier.train()
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:

            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # Adv Domain Discriminator loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_adv_pred = self.adv_discriminator(src_feat_reversed)

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_adv_pred = self.adv_discriminator(trg_feat_reversed)

            # Domain classifier and weights computation
            src_domain_pred = self.domain_classifier(src_feat)
            trg_domain_pred = self.domain_classifier(trg_feat)

            w_s = self.normalize_weight(self.get_src_weights(src_domain_pred, src_pred))
            w_t = self.normalize_weight(self.get_trg_weights(trg_domain_pred, trg_pred))

            src_domain_loss = self.bce(src_domain_pred.squeeze(), domain_label_src)
            trg_domain_loss = self.bce(trg_domain_pred.squeeze(), domain_label_trg)

            src_adv_loss = w_s * F.binary_cross_entropy(src_adv_pred.squeeze(), domain_label_src, reduction='none')
            src_adv_loss = src_adv_loss.mean()
            trg_adv_loss = w_t * F.binary_cross_entropy(trg_adv_pred.squeeze(), domain_label_trg, reduction='none')
            trg_adv_loss = trg_adv_loss.mean()

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss
            adv_loss = src_adv_loss + trg_adv_loss

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                   self.hparams["domain_loss_wt"] * adv_loss

            loss.backward(retain_graph=True)
            domain_loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses = {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                      "Adv Loss": adv_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

    def evaluate(self, test_loader, trg_private_class, src=False):
        self.feature_extractor.eval()
        self.classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = self.feature_extractor(data)
                predictions = self.classifier(features)
                trg_domain_pred = self.domain_classifier(features)
                w_t = self.normalize_weight(self.get_trg_weights(trg_domain_pred, predictions))
                mask = w_t < self.w_0

                if not src:
                    predictions[mask.squeeze()] *= 0

                if self.uniDA:
                    mask = labels >= predictions.shape[-1]
                    labels[mask] = predictions.shape[-1]

                mask = labels < predictions.shape[-1]
                loss = F.cross_entropy(predictions[mask], labels[mask])
                total_loss.append(loss.detach().cpu().item())
                pred = predictions.detach()

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        return loss, full_preds, full_labels

    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred


class OVANet(Algorithm):
    SCENARIO = "UniDA"

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Domain Discriminator
        self.open_set_classifier = classifierOVANet(configs)

        self.optimizer_feature_gen = torch.optim.Adam(
            list(self.feature_extractor.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_clasifier = torch.optim.Adam(
            list(self.open_set_classifier.parameters()) + list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.entropy = Entropy()

    def ova_loss(self, open_preds, label):
        assert len(open_preds.size()) == 3
        assert open_preds.size(1) == 2

        out_open = F.softmax(open_preds, 1)
        label_p = torch.zeros((out_open.size(0),
                               out_open.size(2))).long().to(self.device)
        label_range = torch.arange(0, out_open.size(0)).long()
        label_p[label_range, label] = 1
        label_n = 1 - label_p
        open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                        + 1e-8) * label_p, 1))
        open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] +
                                                        1e-8) * label_n, 1)[0])
        return open_loss_pos, open_loss_neg

    def open_entropy(self, open_preds):
        assert len(open_preds.size()) == 3
        assert open_preds.size(1) == 2
        out_open = F.softmax(open_preds, 1)
        ent_open = torch.mean(torch.mean(torch.sum(-out_open * torch.log(out_open + 1e-8), 1), 1))
        return ent_open

    def entropy(self, p, prob=True, mean=True):
        if prob:
            p = F.softmax(p)
        en = -torch.sum(p * torch.log(p + 1e-5), 1)
        if mean:
            return torch.mean(en)
        else:
            return en

    def update(self, src_loader, trg_loader, avg_meter, val_loader, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs + 1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        with torch.no_grad():
            self.network.eval()
            X = src_loader.dataset.x_data.to(self.device)
            Y = src_loader.dataset.y_data.numpy()
            logits = self.classifier(self.feature_extractor(X))
            preds = logits.detach().cpu().argmax(axis=1).numpy()
        self.network.train()
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:

            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer_clasifier.zero_grad()
            self.optimizer_feature_gen.zero_grad()

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)
            src_open = self.open_set_classifier(src_feat)
            src_open = src_open.view(src_open.size(0), 2, -1)

            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)
            open_loss_pos, open_loss_neg = self.ova_loss(src_open, src_y)
            ## b x 2 x C
            loss_open = 0.5 * (open_loss_pos + open_loss_neg)
            total_loss = loss_open + src_cls_loss

            trg_feat = self.feature_extractor(trg_x)
            trg_open_pred = self.open_set_classifier(trg_feat)
            trg_open_pred = trg_open_pred.view(trg_open_pred.size(0), 2, -1)

            out_open_t = trg_open_pred.view(trg_x.size(0), 2, -1)
            ent_open = self.open_entropy(out_open_t)
            total_loss += ent_open

            total_loss.backward()
            self.optimizer_feature_gen.step()
            self.optimizer_clasifier.step()
            self.optimizer_feature_gen.zero_grad()
            self.optimizer_clasifier.zero_grad()

            losses = {'Total_loss': total_loss.item(), 'Open Loss': loss_open.item(), 'Src_cls_loss': src_cls_loss.item(),
                      "Entropy Open": ent_open.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

    def evaluate(self, test_loader, trg_private_class, src=False):
        feature_extractor = self.feature_extractor.to(self.device)
        classifier = self.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = self.feature_extractor(data)
                predictions = F.softmax(self.classifier(features))
                open_preds = self.open_set_classifier(features)

                conf, pred = predictions.max(dim=1)

                open_preds = F.softmax(open_preds.view(predictions.size(0), 2, -1), 1)
                tmp_range = torch.arange(0, predictions.size(0)).long().to(self.device)
                pred_unk = open_preds[tmp_range, 0, pred]
                ind_unk = np.where(pred_unk.data.cpu().numpy() > 0.5)[0]
                pred[ind_unk] = predictions.shape[-1]
                mask = ind_unk

                if not src:
                    predictions[mask] *= 0

                if self.uniDA:
                    mask = labels >= predictions.shape[-1]
                    labels[mask] = predictions.shape[-1]

                mask = labels < predictions.shape[-1]
                loss = F.cross_entropy(predictions[mask], labels[mask])
                total_loss.append(loss.detach().cpu().item())
                pred = predictions.detach()

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)
        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        return loss, full_preds, full_labels

    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred


class DANCE(Algorithm):
    SCENARIO = "UniDA"

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # hparams
        self.hparams = hparams
        # device
        self.device = device
        self.classifier = classifierNoBias(configs)
        self.rho = np.log(self.configs.num_classes) / 2.0

        self.optimizer_feature_gen = torch.optim.Adam(
            list(self.feature_extractor.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_clasifier = torch.optim.Adam(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.entropy = Entropy()
        self.configs = configs

    def init_memory(self, trg_loader):
        self.ndata = len(trg_loader.dataset.y_data)
        self.lemniscate = LinearAverage(self.configs.feat_dim, self.ndata).to(self.device)

    def entropy(self, p):
        p = F.softmax(p)
        return -torch.mean(torch.sum(p * torch.log(p + 1e-5), 1))

    def entropy_margin(self, p, value, margin=0.2, weight=None):
        p = F.softmax(p)
        return -torch.mean(self.hinge(torch.abs(-torch.sum(p * torch.log(p + 1e-5), 1) - value), margin))

    def hinge(self, input, margin=0.2):
        return torch.clamp(input, min=margin)

    def update(self, src_loader, trg_loader, avg_meter, val_loader, logger):
        self.init_memory(trg_loader)
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs + 1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        with torch.no_grad():
            self.network.eval()
            X = src_loader.dataset.x_data.to(self.device)
            Y = src_loader.dataset.y_data.numpy()
            logits = self.classifier(self.feature_extractor(X))
            preds = logits.detach().cpu().argmax(axis=1).numpy()
            self.network.train()
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y, _ in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            losses = {'Pr_Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y, _), (trg_x, _, trg_index)) in joint_loader:

            src_x, src_y, trg_x, trg_index = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device), trg_index.to(self.device)

            # zero grad
            self.optimizer_clasifier.zero_grad()
            self.optimizer_feature_gen.zero_grad()

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)
            trg_feat = F.normalize(trg_feat)

            feat_mat = self.lemniscate(trg_feat, trg_index)
            feat_mat[:, trg_index] = -1.0
            ### Calculate mini-batch x mini-batch similarity

            feat_mat2 = torch.matmul(trg_feat, trg_feat.t())
            mask = torch.eye(feat_mat2.size(0), feat_mat2.size(0)).bool().to(self.device)
            feat_mat2.masked_fill_(mask, -1)

            loss_nc = self.hparams["eta"] * self.entropy(torch.cat([trg_pred, feat_mat, feat_mat2], 1))
            loss_ent = self.hparams["eta"] * self.entropy_margin(trg_pred, self.rho, self.hparams["margin"])
            total_loss = loss_nc + src_cls_loss + loss_ent

            total_loss.backward()
            self.optimizer_feature_gen.step()
            self.optimizer_clasifier.step()
            self.optimizer_feature_gen.zero_grad()
            self.optimizer_clasifier.zero_grad()

            self.lemniscate.update_weight(trg_feat, trg_index)

            losses = {'Total_loss': total_loss.item(), 'Ent Loss': loss_ent.item(),
                      'Src_cls_loss': src_cls_loss.item(),
                      "Neighbors Clustering ": loss_nc.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

    def evaluate(self, test_loader, trg_private_class, src=False):
        feature_extractor = self.feature_extractor.to(self.device)
        classifier = self.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels, _ in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = self.feature_extractor(data)
                predictions = F.softmax(self.classifier(features))

                entr = -torch.sum(predictions * torch.log(predictions), 1).data.cpu().numpy()

                conf, pred = predictions.max(dim=1)

                pred_unk = np.where(entr > self.rho)
                pred[pred_unk] = predictions.shape[-1]
                mask = pred_unk

                if not src:
                    predictions[mask] *= 0

                if self.uniDA:
                    mask = labels >= predictions.shape[-1]
                    labels[mask] = predictions.shape[-1]
                mask = labels < predictions.shape[-1]
                loss = F.cross_entropy(predictions[mask], labels[mask])
                total_loss.append(loss.detach().cpu().item())
                pred = predictions.detach()

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)
        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        return loss, full_preds, full_labels

    def get_latent_features(self, dataloader):
        feature_set = []
        label_set = []
        logit_set = []
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (data, label, ids) in enumerate(dataloader):
                data = data.to(self.device)
                feature = self.feature_extractor(data)
                logit = self.classifier(feature)
                logit_set.append(logit.cpu())
                feature_set.append(feature.cpu())
                label_set.append(label.cpu())
            feature_set = torch.cat(feature_set, dim=0)
            feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
            logit_set = torch.cat(logit_set, dim=0)
        return feature_set, label_set, logit_set

    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred


class PPOT(Algorithm):
    SCENARIO = "UniDA"

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.optimizer = torch.optim.Adam(                                          
        self.network.parameters(),                                              
        lr=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],                                   
  )
        self.hparams = hparams
        self.configs = configs
        self.device = device

        self.src_prototype = None
        self.num_classes = self.configs.num_classes
        self.alpha = 0
        self.class_weight = 0
        self.beta = 0

        self.softmax = torch.nn.Softmax(dim=1)

    def get_features(self, dataloader):
        feature_set = []
        label_set = []
        self.feature_extractor.eval()
        with torch.no_grad():
            for _, (data, label) in enumerate(dataloader):
                data = data.to(self.device)
                feature = self.feature_extractor(data)
                feature_set.append(feature)
                label_set.append(label)
            feature_set = torch.cat(feature_set, dim=0)
            feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
        return feature_set, label_set

    def get_prototypes(self, dataloader) -> torch.Tensor:
        feature_set, label_set = self.get_features(dataloader)
        class_set = [i for i in range(self.num_classes)]
        source_prototype = torch.zeros((len(class_set), feature_set[0].shape[0]))
        for i in class_set:
            class_feats = feature_set[label_set == i]
            if class_feats.size(0) > 0:
                source_prototype[i] = class_feats.mean(0)
            # else: stays zero (no samples for this class)
        return source_prototype.to(self.device)

    def update_alpha(self, trg_loader) -> np.ndarray:
        num_conf, num_sample = 0, 0
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (trg_x, _) in enumerate(trg_loader):
                trg_x = trg_x.to(self.device)
                output = self.classifier(self.feature_extractor(trg_x))
                output = self.softmax(output)
                conf, _ = output.max(dim=1)
                num_conf += torch.sum(conf > self.hparams["tau1"]).item()
                num_sample += output.shape[0]
            alpha = num_conf / num_sample
            alpha = np.around(alpha, decimals=2)
        return alpha

    def entropy_loss(self, prediction: torch.Tensor, weight=torch.zeros(1)):
        if weight.size(0) == 1:
            entropy = torch.sum(-prediction * torch.log(prediction + 1e-8), 1)
            entropy = torch.mean(entropy)
        else:
            entropy = torch.sum(-prediction * torch.log(prediction + 1e-8), 1)
            entropy = torch.mean(weight * entropy)
        return entropy

    def update(self, src_loader, trg_loader, avg_meter, val_loader, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        nb_pr_epochs = self.hparams['num_epochs_pr']
        for epoch in range(1, nb_pr_epochs + 1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        with torch.no_grad():
            self.network.eval()
            X = src_loader.dataset.x_data.to(self.device)
            Y = src_loader.dataset.y_data.numpy()
            logits = self.classifier(self.feature_extractor(X))
            preds = logits.detach().cpu().argmax(axis=1).numpy()
        self.network.train()

        # Floor alpha at 0.05: partial OT with m < 0.05 + small reg is
        # numerically unstable (Sinkhorn kernel underflows -> NaN). Picking
        # a high tau1 used to drop alpha to 1e-3 and crash the solver.
        self.alpha = max(self.update_alpha(trg_loader), 0.05)
        self.beta = max(self.alpha, 0.05)
        self.class_weight = torch.ones(self.num_classes).to(self.device)
        self.src_prototype = self.get_prototypes(src_loader)

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        self.feature_extractor.train()
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_feat = F.normalize(src_feat, p=2, dim=-1)
            src_pred = self.classifier(src_feat)

            trg_feat = self.feature_extractor(trg_x)
            head = copy.deepcopy(self.classifier)
            for params in list(head.parameters()):
                params.requires_grad = False
            trg_pred = self.softmax(head(trg_feat))
            assert not (torch.isnan(src_feat).any() or torch.isnan(src_pred).any())
            assert not (torch.isnan(trg_feat).any() or torch.isnan(trg_pred).any())

            conf, _ = torch.max(trg_pred, dim=1)

            trg_feat = F.normalize(trg_feat, p=2, dim=-1)
            batch_size = trg_feat.shape[0]

            # update alpha by moving average
            self.alpha = (1 - self.hparams['alpha']) * self.alpha + self.hparams['alpha'] * (conf >= self.hparams['tau1']).sum().item() / conf.size(0)
            self.alpha = max(self.alpha, 0.05)
            # get alpha / beta
            match = self.alpha / self.beta
            assert not np.isnan(match)

            # update source prototype by moving average
            self.src_prototype = self.src_prototype.detach().cpu()
            batch_source_prototype = torch.zeros_like(self.src_prototype)
            for i in range(self.num_classes):
                if (src_y == i).sum().item() > 0:
                    batch_source_prototype[i] = (src_feat[src_y == i].mean(dim=0))
                else:
                    batch_source_prototype[i] = (self.src_prototype[i])
            self.src_prototype = (1 - self.hparams["tau"]) * self.src_prototype + self.hparams["tau"] * batch_source_prototype
            self.src_prototype = F.normalize(self.src_prototype, p=2, dim=-1)
            self.src_prototype = self.src_prototype.to(self.device)

            # get ot loss
            a, b = ot.unif(self.num_classes) + 1e-3, ot.unif(batch_size) + 1e-3
            m = torch.cdist(self.src_prototype, trg_feat) ** 2
            assert not torch.isnan(m).any()
            m_max = m.max().detach()
            m = m / m_max

            pi, log = ot.partial.entropic_partial_wasserstein(a, b, m.detach().cpu().numpy(),
                                                              reg=self.hparams['reg'], m=self.alpha,
                                                              stopThr=1e-10, log=True)
            pi = torch.from_numpy(pi).float().to(self.device)
            assert not torch.isnan(pi).any()
            ot_loss = torch.sqrt(torch.sum(pi * m) * m_max)
            loss = self.hparams['ot'] * ot_loss

            # update class weight and target weight by plan pi
            plan = pi * batch_size
            k = round(self.hparams['neg'] * batch_size)
            min_dist, _ = torch.min(m, dim=0)
            _, indicate = min_dist.topk(k=k, dim=0)
            batch_class_weight = torch.tensor([plan[i, :].sum() for i in range(self.num_classes)]).to(self.device)
            self.class_weight = self.hparams['tau'] * batch_class_weight + (1 - self.hparams['tau']) * self.class_weight
            self.class_weight = self.class_weight * self.num_classes / self.class_weight.sum()
            k_weight = torch.tensor([plan[:, i].sum() for i in range(batch_size)]).to(self.device)
            k_weight /= self.alpha
            u_weight = torch.zeros(batch_size).to(self.device)
            u_weight[indicate] = torch.clamp(1 - k_weight[indicate], min=0)

            # update beta
            self.beta = self.hparams['beta'] * (self.class_weight > self.hparams['tau2']).sum().item() / self.num_classes + (1 - self.hparams['beta']) * self.beta
            self.beta = max(self.beta, 1e-3)

            # get classification loss
            cls_loss = F.cross_entropy(src_pred, src_y, weight=self.class_weight.float())
            loss += cls_loss

            # get entropy loss
            p_ent_loss = self.hparams['p_entropy'] * self.entropy_loss(trg_pred, k_weight)
            n_ent_loss = self.hparams['n_entropy'] * self.entropy_loss(trg_pred, u_weight)
            ent_loss = p_ent_loss - n_ent_loss
            loss += ent_loss

            # compute gradient
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update Prototypes and Alpha
        self.src_prototype = self.get_prototypes(src_loader)
        self.alpha = max(self.update_alpha(trg_loader), 0.05)

        losses = {'Total_loss': loss.item(), 'OT Loss': ot_loss.item(),
                  'Entropic Loss': ent_loss.item(),
                  'Src_cls_loss': cls_loss.item()}

        for key, val in losses.items():
            avg_meter[key].update(val, src_x.size(0))

    def evaluate(self, test_loader, trg_private_class, src=False):
        feature_extractor = self.feature_extractor.to(self.device)
        classifier = self.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = self.feature_extractor(data)
                predictions = self.softmax(self.classifier(features))

                features = self.feature_extractor(data)
                predictions = self.softmax(self.classifier(features))
                confidence, pred = predictions.max(dim=1)
                mask = confidence < self.hparams["thresh"]
                if not src:
                    predictions[mask] *= 0

                if self.uniDA:
                    mask = labels >= predictions.shape[-1]
                    labels[mask] = predictions.shape[-1]

                mask = labels < predictions.shape[-1]
                loss = F.cross_entropy(predictions[mask], labels[mask])
                total_loss.append(loss.detach().cpu().item())
                pred = predictions.detach()

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)
        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        return loss, full_preds, full_labels

    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred


class UniOT(Algorithm):
    SCENARIO = "UniDA"

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # device
        self.configs = configs
        self.device = device
        self.feature_extractor = backbone(configs).to(self.device)
        self.classifier = CLS(configs, temp=hparams['temp']).to(self.device)

        # cluster_head operates on CLS projection output (final_out_channels)
        self.cluster_head = ProtoCLS(configs.final_out_channels, hparams['K'], temp=hparams['temp']).to(self.device)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # hparams
        self.hparams = hparams
        self.nb_classes = configs.num_classes

        # initialize the gamma (coupling in OT) with zeros
        self.gamma = torch.zeros(hparams["batch_size"],
                                 hparams["batch_size"])
        self.gamma.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_feat = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_cls = torch.optim.Adam(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_cluhead = torch.optim.Adam(
            self.cluster_head.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.n_batch = int(hparams['MQ_size'] / hparams['batch_size'])

        # MemoryQueue stores CLS projection output (final_out_channels), not raw backbone features
        self.memqueue = MemoryQueue(configs.final_out_channels, hparams['batch_size'], self.n_batch, hparams['temp']).to(self.device)
        self.beta = None
        self.softmax = torch.nn.Softmax(dim=1)
        self.bce = BCELoss()
        self.t = 0.5

    def init_queue(self, dataloader):
        cnt_i = 0
        while cnt_i < self.n_batch:
            for x, y, id in dataloader:
                x, y, id = x.to(self.device), y.to(self.device), id.to(self.device)
                feats = self.feature_extractor(x)
                proto, preds = self.classifier(feats)
                self.memqueue.update_queue(F.normalize(proto), id)
                cnt_i += 1
                if cnt_i > self.n_batch - 1:
                    break

    def update(self, src_loader, trg_loader, avg_meter, val_loader, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs + 1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        with torch.no_grad():
            self.network.eval()
            X = src_loader.dataset.x_data.to(self.device)
            Y = src_loader.dataset.y_data.numpy()
            _, logits = self.network(X)
            preds = logits.detach().cpu().argmax(axis=1).numpy()
        self.network.train()
        self.init_queue(trg_loader)
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y, _ in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            _, src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            losses = {'Pr_Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        temp = self.hparams['temp']
        for step, ((src_x, src_y, id_source), (trg_x, _, id_target)) in joint_loader:

            if src_x.shape[0] > trg_x.shape[0]:
                src_x = src_x[:trg_x.shape[0]]
                src_y = src_y[:trg_x.shape[0]]
            elif trg_x.shape[0] > src_x.shape[0]:
                trg_x = trg_x[:src_x.shape[0]]

            batch_size = len(src_x)
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            feature_ex_s = self.feature_extractor(src_x)
            feature_ex_t = self.feature_extractor(trg_x)

            before_lincls_feat_s, after_lincls_s = self.classifier(feature_ex_s)
            before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)

            norm_feat_t = F.normalize(before_lincls_feat_t)

            after_cluhead_t = self.cluster_head(before_lincls_feat_t)

            # =====Source Supervision=====
            criterion = nn.CrossEntropyLoss().to(self.device)
            loss_cls = criterion(after_lincls_s, src_y)

            # =====Private Class Discovery=====
            minibatch_size = norm_feat_t.size(0)

            # obtain nearest neighbor from memory queue and current mini-batch
            feat_mat2 = torch.matmul(norm_feat_t, norm_feat_t.t()) / temp
            mask = torch.eye(feat_mat2.size(0), feat_mat2.size(0)).bool().to(self.device)
            feat_mat2.masked_fill_(mask, -1 / temp)

            nb_value_tt, nb_feat_tt = self.memqueue.get_nearest_neighbor(norm_feat_t, id_target.to(self.device))
            neighbor_candidate_sim = torch.cat([nb_value_tt.reshape(-1, 1), feat_mat2], 1)
            values, indices = torch.max(neighbor_candidate_sim, 1)
            neighbor_norm_feat = torch.zeros((minibatch_size, norm_feat_t.shape[1])).to(self.device)
            for i in range(minibatch_size):
                neighbor_candidate_feat = torch.cat([nb_feat_tt[i].reshape(1, -1), norm_feat_t], 0)
                neighbor_norm_feat[i, :] = neighbor_candidate_feat[indices[i], :]

            neighbor_output = self.cluster_head(neighbor_norm_feat)

            # fill input features with memory queue
            fill_size_ot = self.hparams['K']
            mqfill_feat_t = self.memqueue.random_sample(fill_size_ot)
            mqfill_output_t = self.cluster_head(mqfill_feat_t)

            # OT process
            # mini-batch feat (anchor) | neighbor feat | filled feat (sampled from memory queue)
            S_tt = torch.cat([after_cluhead_t, neighbor_output, mqfill_output_t], 0)
            S_tt *= temp
            # Paper Sec. 3.2 / Algo. 1: Sinkhorn ε for the SwAV-style PCD
            # assignment is 0.05 in their public code (changwxx/UniOT-for-UniDA);
            # keep that value here. Larger ε washes out the cluster contrast.
            Q_tt = sinkhorn(S_tt.detach(), epsilon=0.05, sinkhorn_iterations=3)
            Q_tt_tilde = Q_tt * Q_tt.size(0)
            anchor_Q = Q_tt_tilde[:minibatch_size, :]
            neighbor_Q = Q_tt_tilde[minibatch_size:2 * minibatch_size, :]

            # compute loss_PCD
            loss_local = 0
            for i in range(minibatch_size):
                sub_loss_local = 0
                sub_loss_local += -torch.sum(neighbor_Q[i, :] * F.log_softmax(after_cluhead_t[i, :]))
                sub_loss_local += -torch.sum(anchor_Q[i, :] * F.log_softmax(neighbor_output[i, :]))
                sub_loss_local /= 2
                loss_local += sub_loss_local
            loss_local /= minibatch_size
            loss_global = -torch.mean(torch.sum(anchor_Q * F.log_softmax(after_cluhead_t, dim=1), dim=1))
            loss_PCD = (loss_global + loss_local) / 2

            # =====Common Class Detection=====
            source_prototype = self.classifier.ProtoCLS.fc.weight
            if self.beta is None:
                self.beta = ot.unif(source_prototype.size()[0])

            # fill input features with memory queue
            fill_size_uot = self.n_batch * batch_size
            mqfill_feat_t = self.memqueue.random_sample(fill_size_uot)
            ubot_feature_t = torch.cat([mqfill_feat_t, norm_feat_t], 0)

            # Adaptive filling
            newsim, fake_size = adaptive_filling(ubot_feature_t, source_prototype, self.hparams['gamma'], self.beta, fill_size_uot)

            # UOT-based CCD
            high_conf_label_id, high_conf_label, _, new_beta = ubot_CCD(newsim, self.beta, fake_size=fake_size,
                                                                        fill_size=fill_size_uot, mode='minibatch')
            # adaptive update for marginal probability vector
            self.beta = self.hparams['mu'] * self.beta + (1 - self.hparams['mu']) * new_beta

            # fix the bug raised in https://github.com/changwxx/UniOT-for-UniDA/issues/1
            if high_conf_label_id.size(0) > 0:
                loss_CCD = criterion(after_lincls_t[high_conf_label_id, :], high_conf_label[high_conf_label_id])
            else:
                loss_CCD = 0

            loss_all = loss_cls + self.hparams['lam'] * (loss_CCD)

            self.optimizer_feat.zero_grad()
            self.optimizer_cls.zero_grad()
            self.optimizer_cluhead.zero_grad()
            loss_all.backward()
            self.optimizer_feat.step()
            self.optimizer_cls.step()
            self.optimizer_cluhead.step()

            self.classifier.ProtoCLS.weight_norm()  # very important for proto-classifier
            self.cluster_head.weight_norm()  # very important for proto-classifier
            self.memqueue.update_queue(norm_feat_t, id_target.to(self.device))

            losses = {'Total_loss': loss_all.item(), 'loss_cls': loss_cls.item(),
                      'loss_PCD': loss_PCD.item(),
                      'loss_CCD': loss_CCD}

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

    def evaluate(self, test_loader, trg_private_class, src=False):
        feature_extractor = self.feature_extractor.to(self.device)
        classifier = self.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []
        norm_feat_t_list = []

        with torch.no_grad():
            for data, labels, id in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = feature_extractor(data)
                before_lincls_feat_t, predictions = classifier(features)
                norm_feat_t = F.normalize(before_lincls_feat_t)

                if self.uniDA and not src:
                    mask = labels < predictions.shape[-1]
                    loss = F.cross_entropy(predictions[mask], labels[mask])
                    total_loss.append(loss.detach().cpu().item())
                else:
                    loss = F.cross_entropy(predictions, labels)
                    total_loss.append(loss.detach().cpu().item())
                pred = predictions.detach()

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)
                norm_feat_t_list.append(norm_feat_t)
        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        norm_feat_t = torch.cat((norm_feat_t_list))

        source_prototype = classifier.ProtoCLS.fc.weight

        stopThr = 1e-6
        # Adaptive filling
        newsim, fake_size = adaptive_filling(norm_feat_t.to(self.device),
                                             source_prototype, self.hparams['gamma'], self.beta, 0, stopThr=stopThr)

        # obtain predict label
        _, __, pred_label, ___ = ubot_CCD(newsim, self.beta, fake_size=fake_size, fill_size=0, mode='minibatch',
                                          stopThr=stopThr)
        pred_label = pred_label.cpu().data.numpy()
        # CCD-based unknown rejection only applies to the target domain.
        # On the source loader (src=True) all samples are by construction
        # known; running the rejection mask there zeros out legitimate
        # source predictions and corrupts source-domain accuracy reporting.
        if not src:
            mask = pred_label == self.nb_classes
            full_preds[mask] *= 0

        return loss, full_preds, full_labels

    def get_latent_features(self, dataloader):
        feature_set = []
        label_set = []
        logits_set = []
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (data, label, id) in enumerate(dataloader):
                data = data.to(self.device)
                feature = self.feature_extractor(data)
                _, logit = self.classifier(feature)
                feature_set.append(feature.cpu())
                label_set.append(label.cpu())
                logits_set.append(logit.cpu())
            feature_set = torch.cat(feature_set, dim=0)
            feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
            logits_set = torch.cat(logits_set, dim=0)
        return feature_set, label_set, logits_set

    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred


class UniJDOT(Algorithm):
    SCENARIO = "UniDA"

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # device
        self.device = device
        self.feature_extractor = backbone(configs).to(self.device)
        self.classifier = CLS(configs).to(self.device)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # hparams
        self.hparams = hparams
        self.nb_classes = configs.num_classes

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_feat = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_cls = torch.optim.Adam(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.beta = None
        self.softmax = torch.nn.Softmax(dim=1)
        self.bce = BCELoss()
        self.src_latent_cluster = None
        self.register_buffer("final_threshold", torch.tensor(0.0))

        self.final_threshold = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

        self.threshold_method = self.get_thresholding_method()
        # ClassMemoryQueue stores CLS projection output (final_out_channels)
        self.memqueue_feat = ClassMemoryQueue(configs.final_out_channels, self.nb_classes, hparams['n_batch']).to(self.device)

    def init_queue(self, dataloader):
        cnt_i = 0
        for x, y, id in dataloader:
            x, y, id = x.to(self.device), y.to(self.device), id.to(self.device)
            feature_ex_s = self.feature_extractor(x)
            before_lincls_feat_s, after_lincls_s = self.classifier(feature_ex_s)
            self.memqueue_feat.update_queue(F.normalize(before_lincls_feat_s), y)
            cnt_i += 1
            if self.memqueue_feat.is_memory_full().all():
                break

    def infomax_loss(self, cluster_assignments, eps=1e-10):
        """
        InfoMax loss function to maximize mutual information between inputs and cluster assignments.

        Args:
            cluster_assignments (torch.Tensor): Tensor of shape (N, K) representing the probability distribution
                                                over K clusters for each sample.
            eps (float): Small constant to avoid numerical issues with log.

        Returns:
            torch.Tensor: Scalar loss value (InfoMax loss).
        """
        # Batch size and number of clusters
        N, K = cluster_assignments.shape

        # Step 1: Compute the marginal distribution p(z) over clusters (averaging over samples)
        marginal_prob = cluster_assignments.mean(dim=0)  # Shape (K,)

        # Step 2: Compute H(Z) - Entropy of the marginal distribution (cluster assignments)
        H_Z = -torch.sum(marginal_prob * torch.log(marginal_prob + eps))

        # Step 3: Compute H(Z|X) - Conditional entropy of cluster assignment given input
        H_Z_given_X = -torch.sum(cluster_assignments * torch.log(cluster_assignments + eps)) / N

        # Step 4: InfoMax loss is H(Z) - H(Z|X)
        infomax_loss_value = H_Z - H_Z_given_X

        return infomax_loss_value

    def get_thresholding_method(self):
        return getattr(sfil, self.hparams['threshold_method'])

    def class_centroids(self, x, y):
        # Reduce the sum of each feature across samples within a class
        class_sums = torch.einsum("ji,jk->ki", x, y.float().to(self.device))
        # Count the number of samples in each class (sum along the sample dimension)
        class_counts = torch.sum(y, dim=0)

        # Avoid division by zero for empty classes
        class_counts[class_counts == 0] = 1

        # Divide class sums by class counts to get centroids
        centroids = (class_sums.T / class_counts).T

        return centroids

    def ini_centroids(self, src_dl):
        """Compute source class centroids from normalized features."""
        all_centroids = []
        with TrainingModeManager([self.feature_extractor, self.classifier], train=False) as mgr, \
                torch.no_grad():
            for im_s, label_source, id_s in src_dl:
                im_s = im_s.to(self.device)
                label_source = label_source.to(self.device)
                feature_ex_s = self.feature_extractor(im_s)
                before_lincls_feat_s, _ = self.classifier(feature_ex_s)
                norm_feat_s = F.normalize(before_lincls_feat_s)
                y_src = torch.eye(self.nb_classes, dtype=torch.int8, device=self.device)[label_source]
                ctr = self.class_centroids(norm_feat_s, y_src)
                all_centroids.append(ctr.cpu())
        return torch.stack(all_centroids).mean(dim=0)

    def centroids_target(self, trg_dl, K):
        X = []
        with TrainingModeManager([self.feature_extractor], train=False) as mgr, \
                torch.no_grad():
            for im_t, label_target, id_t in trg_dl:
                im_t = im_t.to(self.device)
                feature_ex_t = self.feature_extractor.forward(im_t)
                before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)
                norm_feat_t = F.normalize(before_lincls_feat_t)
                X.append(norm_feat_t)
        X = torch.cat((X), 0).cpu()

        kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(X)
        return kmeans.cluster_centers_

    def update_centroids_target(self, X, cen):
        X = X.cpu()
        cen = cen.cpu()
        dd = torch.cdist(X, cen)
        pred_cen = dd.argmin(axis=1)

        for k in range(cen.shape[0]):
            ix = pred_cen == k
            if ix.any():
                cen[k] = 0.9 * cen[k] + 0.1 * X[ix].mean(axis=0)

        return cen

    def update(self, src_loader, trg_loader, avg_meter, val_loader, logger):
        self.src_loader = src_loader
        self.init_queue(src_loader)
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        self.src_latent_cluster = self.ini_centroids(src_loader).to(self.device)
        self.trg_latent_cluster = self.centroids_target(trg_loader, self.hparams['K'])
        self.trg_latent_cluster = torch.from_numpy(self.trg_latent_cluster).to(torch.float)

        nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs + 1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        with torch.no_grad():
            self.network.eval()
            X = self.src_loader.dataset.x_data.to(self.device)
            Y = self.src_loader.dataset.y_data.numpy()
            _, logits = self.network(X)
            preds = logits.detach().cpu().argmax(axis=1).numpy()
        self.network.train()
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        # UniDABench parity: truncate target buffer to trg_mem_size before
        # computing the final threshold. Threshold-on-full-target diverged
        # from the reference and gave more degenerate yen histograms.
        trg_mem_size = self.hparams['trg_mem_size']
        self.trg_feats_mem = []
        self.trg_preds_mem = []
        cnt_i = 0
        with torch.no_grad():
            self.network.eval()
            for x, y, id in trg_loader:
                x, y, id = x.to(self.device), y.to(self.device), id.to(self.device)
                feature_ex_t = self.feature_extractor(x)
                before_lincls_feat_t, after_lincls_s = self.classifier(feature_ex_t)
                norm_feat_t = F.normalize(before_lincls_feat_t)
                self.trg_feats_mem.append(norm_feat_t)
                self.trg_preds_mem.append(after_lincls_s)
                cnt_i += after_lincls_s.shape[0]
                if cnt_i > trg_mem_size:
                    break
        self.trg_feats_mem = torch.concatenate(self.trg_feats_mem)[:trg_mem_size]
        self.trg_preds_mem = torch.concatenate(self.trg_preds_mem)[:trg_mem_size]

        if self.hparams['joint_decision']:
            dist_trg_tr = self.compute_cluster_distance(self.trg_feats_mem)
            soft_trg_tr = self.joint_decision(self.trg_preds_mem, dist_trg_tr)
        else:
            soft_trg_tr = F.softmax(self.trg_preds_mem, dim=1)
        conf, preds = soft_trg_tr.max(dim=1)
        new_value = self.threshold_method(conf.detach().cpu().numpy())
        self.final_threshold.data.fill_(new_value)

        print(f"\n[UniJDOT Debug] final_threshold = {new_value:.4f} "
              f"conf mean={conf.mean():.4f} std={conf.std():.4f} "
              f"reject={(conf < new_value).float().mean():.2%}")

        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y, _ in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            _, src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            losses = {'Pr_Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

    def compute_cluster_distance(self, x_test):
        x = self.memqueue_feat.mem_feat

        x, x_test = x.squeeze(), x_test.squeeze()
        if len(x_test.shape) == 1:
            x_test = x_test.unsqueeze(0)
        res = self.memqueue_feat.compute_distances(x_test)

        d = -1 * res
        d = F.softmax(d, dim=1)

        return d

    def compute_cluster_distance2(self, x_test):
        x = self.src_loader.dataset.x_data
        y = self.src_loader.dataset.y_data
        x, x_test, y = torch.Tensor(x).to(self.device), torch.Tensor(x_test), torch.Tensor(y).to(self.device).long()
        feature_ex_t = self.feature_extractor(x)
        before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)

        x = F.normalize(before_lincls_feat_t)
        x, x_test = x.squeeze(), x_test.squeeze()
        nb_classes = self.nb_classes
        res = torch.empty(x_test.shape[0], nb_classes)
        res = torch.zeros_like(res)
        for i, ll in enumerate(range(nb_classes)):
            dist = torch.cdist(x[y == ll], x_test).min(axis=0).values
            res[:, int(ll)] = dist

        d = -1 * res
        d = F.softmax(d, dim=1)
        return d.max(axis=0).values

    def joint_decision(self, preds, distance):
        """Paper eq: p'_t = σ(h(x_t) · σ(-d_t))"""
        preds = preds.to(self.device) if not preds.is_cuda else preds
        distance = distance.to(self.device)
        return F.softmax(preds * distance, dim=1)

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        for step, ((src_x, src_y, id_source), (trg_x, _, id_target)) in joint_loader:

            if src_x.shape[0] > trg_x.shape[0]:
                src_x = src_x[:trg_x.shape[0]]
                src_y = src_y[:trg_x.shape[0]]
            elif trg_x.shape[0] > src_x.shape[0]:
                trg_x = trg_x[:src_x.shape[0]]

            batch_size = len(src_x)
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            feature_ex_s = self.feature_extractor(src_x)
            feature_ex_t = self.feature_extractor(trg_x)

            before_lincls_feat_s, after_lincls_s = self.classifier(feature_ex_s)
            before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)

            norm_feat_s = F.normalize(before_lincls_feat_s)
            norm_feat_t = F.normalize(before_lincls_feat_t)

            # =====Source Supervision=====

            y_src = torch.eye(after_lincls_s.shape[-1], dtype=torch.int8).to(self.device)[src_y]
            self.memqueue_feat.update_queue(norm_feat_s, src_y.to(self.device))

            centr = self.class_centroids(norm_feat_s, y_src).detach().cpu()
            self.src_latent_cluster = 0.9 * self.src_latent_cluster + 0.1 * centr.to(self.device)
            # TRG Centroids
            self.trg_latent_cluster = self.update_centroids_target(norm_feat_t.detach().cpu(), self.trg_latent_cluster)
            # Centroids are in CLS projection space (128-dim), pass directly to ProtoCLS
            trg_cen_dev = self.trg_latent_cluster.to(self.device)
            after_lincls_cen = self.classifier.ProtoCLS(trg_cen_dev)
            trg_soft_cen = torch.nn.functional.softmax(after_lincls_cen).double()

            if self.hparams['joint_decision']:
                dist = self.compute_cluster_distance(norm_feat_t)
                soft_t = self.joint_decision(after_lincls_t, dist)
            else:
                soft_t = F.softmax(after_lincls_t, dim=1)
            conf, preds = soft_t.max(dim=1)
            threshold = self.threshold_method(conf.detach().cpu().numpy())
            mask = (conf < threshold).to(self.device)

            C0 = torch.zeros((len(norm_feat_s) + len(trg_cen_dev), len(norm_feat_t))).to(self.device)

            # nonOOD SRC
            C0[:len(norm_feat_s), ~mask] = torch.cdist(norm_feat_s, norm_feat_t[~mask])
            # OOD Dummy
            C0[len(norm_feat_s):, mask] = torch.cdist(trg_cen_dev, norm_feat_t[mask])

            maxc = torch.max(C0).item()
            # ODD SRC
            C0[:len(norm_feat_s), mask] = maxc
            # nonOOD Dummy
            C0[len(norm_feat_s):, ~mask] = maxc

            C1 = torch.zeros(C0.shape).to(self.device)
            C_preds = torch.ones((trg_cen_dev.shape[0], y_src.shape[-1])) / self.nb_classes
            C_preds = C_preds.to(self.device)

            # nonOOD SRC
            C1[:len(norm_feat_s), ~mask] = torch.cdist(y_src.float(), F.softmax(after_lincls_t, dim=1)[~mask])
            # OOD Dummy
            C1[len(norm_feat_s):, mask] = torch.cdist(C_preds, F.softmax(after_lincls_t, dim=1)[mask])
            maxc = torch.max(C1).item()
            # ODD SRC
            C1[:len(norm_feat_s), mask] = maxc
            # nonOOD Dummy
            C1[len(norm_feat_s):, ~mask] = maxc

            C = (self.hparams['alpha'] * C0 + self.hparams['lamb'] * C1)

            with torch.no_grad():

                a, b = ot.unif(C.size(0)), ot.unif(C.size(1))
                ratio = (mask.sum() / len(mask)).detach().cpu().item()
                a[:len(norm_feat_s)] = 0.5 / len(a[:len(norm_feat_s)])
                a[len(norm_feat_s):] = 0.5 / len(a[len(norm_feat_s):])

                gamma = ot.unbalanced.mm_unbalanced(
                    a, b, C.detach().cpu().numpy(), reg_m=0.5)
                gamma = torch.tensor(gamma).float().to(self.device)

            assert not torch.isnan(gamma).any()
            assert not torch.isnan(feature_ex_t).any()
            assert not torch.isnan(feature_ex_s).any()
            assert not torch.isnan(after_lincls_t).any()
            assert not torch.isnan(after_lincls_s).any()

            label_align_loss = (C * gamma)

            label_align_loss_nonOOD = label_align_loss[:len(norm_feat_s), ~mask].sum()
            label_align_loss_OOD = label_align_loss[len(norm_feat_s):, mask].sum()

            label_align_loss = (label_align_loss_nonOOD + label_align_loss_OOD) / 2

            criterion = nn.CrossEntropyLoss().to(self.device)
            loss_cls = self.hparams['src_weight'] * criterion(after_lincls_s, src_y)

            loss_all = loss_cls + label_align_loss

            self.optimizer_feat.zero_grad()
            self.optimizer_cls.zero_grad()
            loss_all.backward()
            self.optimizer_feat.step()
            self.optimizer_cls.step()

            self.classifier.ProtoCLS.weight_norm()  # very important for proto-classifier

            losses = {'Total_loss': loss_all.item(), 'loss_cls': loss_cls.item(),
                      'loss_align': label_align_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

    def evaluate(self, test_loader, trg_private_class, src=False):
        self.feature_extractor.eval()
        self.classifier.eval()

        total_loss, preds_list, labels_list = [], [], []
        all_confs, all_argmax, all_rejected = [], [], []

        with torch.no_grad():
            for data, labels, ids in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = self.feature_extractor(data)
                before_lincls_feat_t, predictions = self.classifier(features)
                if self.hparams['joint_decision']:
                    norm_feat_t = F.normalize(before_lincls_feat_t)
                    dist = self.compute_cluster_distance(norm_feat_t)
                    soft = self.joint_decision(predictions, dist)
                else:
                    soft = F.softmax(predictions, dim=1)

                # Store the same distribution we thresholded on so that
                # decision_function picks class labels consistent with joint_decision.
                output = soft.clone()

                conf, preds = soft.max(dim=1)
                # UniDABench parity: single-threshold rejection on confidence.
                reject_mask = conf < self.final_threshold
                if not src:
                    all_confs.append(conf.cpu())
                    all_argmax.append(preds.cpu())
                    all_rejected.append(reject_mask.cpu())
                    output[reject_mask.squeeze()] *= 0

                if self.uniDA:
                    lbl_mask = labels >= predictions.shape[-1]
                    labels[lbl_mask] = predictions.shape[-1]

                kn_mask = labels < predictions.shape[-1]
                loss = F.cross_entropy(predictions[kn_mask], labels[kn_mask])
                total_loss.append(loss.detach().cpu().item())
                pred = output.detach()

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        if not src and all_confs:
            all_confs = torch.cat(all_confs)
            all_argmax = torch.cat(all_argmax)
            all_rejected = torch.cat(all_rejected)
            full_labels_dbg = torch.cat(labels_list).cpu()
            num_cls = preds_list[0].shape[-1]
            print(f"[UniJDOT Eval ALL] n={len(all_confs)} threshold={self.final_threshold.item():.4f}")
            print(f"[UniJDOT Eval ALL] conf: min={all_confs.min():.4f} max={all_confs.max():.4f} "
                  f"mean={all_confs.mean():.4f} std={all_confs.std():.4f}")
            print(f"[UniJDOT Eval ALL] rejected: {all_rejected.float().mean():.2%}")
            print(f"[UniJDOT Eval ALL] pred class dist (pre-reject): "
                  f"{torch.bincount(all_argmax, minlength=num_cls).tolist()}")
            print(f"[UniJDOT Eval ALL] true label dist: "
                  f"{torch.bincount(full_labels_dbg.clamp(min=0, max=num_cls), minlength=num_cls+1).tolist()} "
                  f"(last bin = unknown)")

        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        return loss, full_preds, full_labels

    def get_latent_features(self, dataloader):
        feature_set = []
        label_set = []
        logits_set = []
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (data, label, ids) in enumerate(dataloader):
                data = data.to(self.device)
                feature = self.feature_extractor(data)
                _, logit = self.classifier(feature)
                feature_set.append(feature.cpu())
                label_set.append(label.cpu())
                logits_set.append(logit.cpu())
            feature_set = torch.cat(feature_set, dim=0)
            feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
            logits_set = torch.cat(logits_set, dim=0)
        return feature_set, label_set, logits_set

    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred


class RAINCOAT(Algorithm):
    """
    Raincoat: Domain Adaptation for Time Series Under Feature and Label Shifts.
    He et al., ICML 2023.

    Uses a dual time-frequency encoder with Sinkhorn OT alignment and
    decoder-based correction. Unknown detection via drift-based approach:
    drift = |d(z_pre, w_c) - d(z_post, w_c)| with DIP bimodality test + 2-means.

    Training phases:
        Phase 1 (Alignment): source CE + Sinkhorn OT + reconstruction
        Phase 2 (Correction): target-only reconstruction
    Inference:
        Phase 3 (Drift Detection): per-class drift thresholding
    """
    SCENARIO = "UniDA"

    def __init__(self, backbone, configs, hparams, device):
        # Skip Algorithm.__init__ — Raincoat uses its own encoder, not the standard backbone
        torch.nn.Module.__init__(self)
        self.configs = configs
        self.hparams = hparams
        self.device = device
        self.cross_entropy = nn.CrossEntropyLoss()

        # Add Raincoat-specific fields to configs from hparams
        configs.kernel_size = hparams.get('kernel_size', 5)
        configs.fourier_modes = hparams.get('fourier_modes', configs.sequence_len // 2)
        configs.out_dim = configs.fourier_modes * 2 + configs.final_out_channels * configs.features_len

        # Raincoat uses its own TFEncoder instead of the standard backbone
        self.feature_extractor = TFEncoder(configs)
        self.decoder = TFDecoder(configs)
        self.classifier = RaincoatClassifier(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # Sinkhorn OT for alignment
        self.sink = SinkhornDistance(eps=hparams.get('sinkhorn_eps', 1e-3), max_iter=1000, reduction='sum')
        self.recons = nn.L1Loss(reduction='sum')

        # Phase 1 optimizer: all parameters
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.decoder.parameters()) +
            list(self.classifier.parameters()),
            lr=hparams['learning_rate'],
            weight_decay=hparams['weight_decay'],
        )
        # Phase 2 optimizer: encoder + decoder only (classifier frozen)
        self.coptimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.decoder.parameters()),
            lr=hparams['learning_rate'],
            weight_decay=hparams['weight_decay'],
        )

        # Drift detection state
        self.pre_correction_state = None
        self.drift_thresholds = {}

    def update(self, src_loader, trg_loader, avg_meter, val_loader, logger):
        best_src_risk = float('inf')
        best_model = None

        num_epochs = self.hparams["num_epochs"]
        num_epochs_correct = self.hparams.get("num_epochs_correct", num_epochs)
        # num_epochs = 50
        # num_epochs_correct = 50

        # ---- Phase 1: Alignment ----
        for epoch in range(1, num_epochs + 1):
            self.train()
            joint_loader = zip(src_loader, trg_loader)

            for (src_x, src_y, *_), (trg_x, *_) in joint_loader:
                src_x = src_x.float().to(self.device)
                src_y = src_y.long().to(self.device)
                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()

                src_feat, out_s = self.feature_extractor(src_x)
                trg_feat, out_t = self.feature_extractor(trg_x)

                # Reconstruction loss
                src_recon = self.decoder(src_feat, out_s)
                trg_recon = self.decoder(trg_feat, out_t)
                recons = 1e-4 * (self.recons(src_recon, src_x) + self.recons(trg_recon, trg_x))
                recons.backward(retain_graph=True)

                # Sinkhorn OT alignment
                dr, _, _ = self.sink(src_feat, trg_feat)
                dr.backward(retain_graph=True)

                # Source classification
                src_pred = self.classifier(src_feat)
                loss_cls = self.cross_entropy(src_pred, src_y)
                loss_cls.backward(retain_graph=True)

                self.optimizer.step()

                avg_meter['Src_cls_loss'].update(loss_cls.item(), src_x.size(0))
                avg_meter['Sink'].update(dr.item(), src_x.size(0))
                avg_meter['Recon'].update(recons.item(), src_x.size(0))

            # Save best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.state_dict())

            logger.debug(f'[Align Epoch : {epoch}/{num_epochs}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        # ---- Save pre-correction encoder state ----
        self.pre_correction_state = deepcopy(self.feature_extractor.state_dict())

        # ---- Phase 2: Correction (source + target reconstruction) ----
        for epoch in range(1, num_epochs_correct + 1):
            self.train()
            for (src_x, *_), (trg_x, *_) in zip(src_loader, trg_loader):
                src_x = src_x.float().to(self.device)
                trg_x = trg_x.float().to(self.device)

                self.coptimizer.zero_grad()
                src_feat, out_s = self.feature_extractor(src_x)
                trg_feat, out_t = self.feature_extractor(trg_x)
                src_recon = self.decoder(src_feat, out_s)
                trg_recon = self.decoder(trg_feat, out_t)
                recons = 1e-4 * (self.recons(src_recon, src_x) + self.recons(trg_recon, trg_x))
                recons.backward()
                self.coptimizer.step()

                avg_meter['Recon_correct'].update(recons.item(), trg_x.size(0))

            logger.debug(f'[Correct Epoch : {epoch}/{num_epochs_correct}]')
            logger.debug(f'Recon_correct\t: {avg_meter["Recon_correct"].avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.state_dict()
        if best_model is None:
            best_model = deepcopy(last_model)

        return last_model, best_model

    def _extract_features_and_preds(self, loader):
        """Extract features and predicted classes for all samples in loader."""
        self.eval()
        all_features, all_preds = [], []
        with torch.no_grad():
            for data, *_ in loader:
                data = data.float().to(self.device)
                features, _ = self.feature_extractor(data)
                logits = self.classifier(features)
                preds = logits.argmax(dim=1)
                all_features.append(features.cpu())
                all_preds.append(preds.cpu())
        return torch.cat(all_features, dim=0), torch.cat(all_preds, dim=0)

    def _compute_proto_distances(self, features, pred_classes):
        """Cosine similarity between each sample and its assigned class prototype."""
        W = self.classifier.logits.weight.detach().cpu()
        W_norm = F.normalize(W, dim=1)
        features_norm = F.normalize(features, dim=1)
        assigned_protos = W_norm[pred_classes]
        distances = (features_norm * assigned_protos).sum(dim=1)
        return distances

    def _run_drift_detection(self, test_loader, dip_p_threshold=0.05):
        """
        Stage 3: Drift-based unknown detection.
        1. Post-correction features z_c -> d(z_c, w_c)
        2. Pre-correction features z_a -> d(z_a, w_c)
        3. drift = |d_a - d_c| per sample
        4. Per-class: DIP test -> if bimodal, 2-means -> higher centroid cluster = unknown
        """
        assert self.pre_correction_state is not None, \
            "Pre-correction state not saved. Run update() first."

        # Save post-correction state
        post_correction_state = deepcopy(self.feature_extractor.state_dict())

        # Post-correction features
        z_c, pred_classes = self._extract_features_and_preds(test_loader)
        d_c = self._compute_proto_distances(z_c, pred_classes)

        # Pre-correction features (same class assignments from post-correction)
        self.feature_extractor.load_state_dict(self.pre_correction_state)
        z_a, _ = self._extract_features_and_preds(test_loader)
        d_a = self._compute_proto_distances(z_a, pred_classes)

        # Restore post-correction state
        self.feature_extractor.load_state_dict(post_correction_state)

        # Drift
        drift = torch.abs(d_a - d_c).numpy()
        pred_np = pred_classes.numpy()
        is_unknown = np.zeros(len(drift), dtype=bool)

        num_classes = self.classifier.logits.weight.shape[0]
        self.drift_thresholds = {}

        for c in range(num_classes):
            class_mask = pred_np == c
            if class_mask.sum() < 2:
                continue

            class_drift = drift[class_mask]
            class_indices = np.where(class_mask)[0]

            # DIP bimodality test
            is_bimodal = False
            if HAS_DIPTEST:
                _, p_value = dip_test(class_drift)
                is_bimodal = p_value < dip_p_threshold
            else:
                is_bimodal = True

            if is_bimodal and class_mask.sum() >= 2:
                km = KMeans(n_clusters=2, random_state=42, n_init=10)
                labels = km.fit_predict(class_drift.reshape(-1, 1))
                centroids = km.cluster_centers_.flatten()

                unknown_cluster = np.argmax(centroids)
                unknown_mask_local = labels == unknown_cluster

                self.drift_thresholds[c] = float(np.mean(centroids))

                for idx, is_unk in zip(class_indices, unknown_mask_local):
                    is_unknown[idx] = is_unk
            else:
                self.drift_thresholds[c] = float(np.max(class_drift)) + 1.0

        return pred_np, is_unknown

    def evaluate(self, test_loader, trg_private_class, src=False):
        self.feature_extractor.eval()
        self.classifier.eval()

        # For source evaluation or if no pre-correction state, use standard forward pass
        if src or self.pre_correction_state is None:
            total_loss, preds_list, labels_list = [], [], []
            with torch.no_grad():
                for data, labels, *_ in test_loader:
                    data = data.float().to(self.device)
                    labels = labels.view((-1)).long().to(self.device)

                    features, _ = self.feature_extractor(data)
                    predictions = self.classifier(features)

                    mask = labels < predictions.shape[-1]
                    loss = F.cross_entropy(predictions[mask], labels[mask])
                    total_loss.append(loss.detach().cpu().item())

                    preds_list.append(predictions.detach())
                    labels_list.append(labels)

            loss = torch.tensor(total_loss).mean()
            full_preds = torch.cat(preds_list)
            full_labels = torch.cat(labels_list)
            return loss, full_preds, full_labels

        # Target evaluation with drift-based unknown detection
        dip_p = self.hparams.get('dip_p_threshold', 0.05)
        pred_classes, is_unknown = self._run_drift_detection(test_loader, dip_p)

        # Build predictions tensor: unknown samples get zeroed-out rows
        total_loss, preds_list, labels_list = [], [], []
        with torch.no_grad():
            for data, labels, *_ in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                features, _ = self.feature_extractor(data)
                predictions = self.classifier(features)

                mask = labels < predictions.shape[-1]
                loss = F.cross_entropy(predictions[mask], labels[mask])
                total_loss.append(loss.detach().cpu().item())

                preds_list.append(predictions.detach())
                labels_list.append(labels)

        loss = torch.tensor(total_loss).mean()
        full_preds = torch.cat(preds_list)
        full_labels = torch.cat(labels_list)

        # Zero out predictions for samples detected as unknown
        # (decision_function maps zeroed rows to -1)
        for i in range(len(is_unknown)):
            if is_unknown[i]:
                full_preds[i] *= 0

        return loss, full_preds, full_labels

    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred


class OSBP(Algorithm):
    """
    Open Set Back-Propagation (Saito et al., ECCV 2018).

    Native open-set DA with K+1 classifier head (K known + 1 unknown).
    Uses gradient reversal to push target samples toward the decision boundary
    between known and unknown classes. The adversarial target loss collapses
    K+1 softmax outputs into binary (known_sum, unknown) and trains against
    a uniform (0.5, 0.5) target via BCE.

    This is a NATIVE method: the unknown class is explicitly modeled during
    training, unlike post-hoc threshold-based approaches.
    """
    SCENARIO = "OSDA"

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)
        self.hparams = hparams
        self.device = device
        self.num_classes = configs.num_classes

        # Replace standard classifier with K+1 classifier (includes bottleneck)
        self.classifier = classifierOSBP(configs).to(self.device)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # SGD with momentum/nesterov (matching original paper exactly)
        self.optimizer_g = torch.optim.SGD(
            self.feature_extractor.parameters(),
            lr=hparams["learning_rate"],
            momentum=0.9, weight_decay=5e-4, nesterov=True,
        )
        self.optimizer_c = torch.optim.SGD(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            momentum=0.9, weight_decay=5e-4, nesterov=True,
        )
        self.optimizer = self.optimizer_g  # for pretrain_epoch compatibility

    def update(self, src_loader, trg_loader, avg_meter, val_loader, logger):
        best_src_risk = float('inf')
        best_model = None

        # Phase 1: Pretrain on source only
        nb_pr_epochs = self.hparams.get("num_epochs_pr", 0)
        for epoch in range(1, nb_pr_epochs + 1):
            self.pretrain_epoch_osbp(src_loader, avg_meter)
            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        # Phase 2: Adversarial training
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()
        return last_model, best_model

    def pretrain_epoch_osbp(self, src_loader, avg_meter):
        """Source-only pretraining with K+1 classifier (only K classes get gradients)."""
        self.train()
        for src_x, src_y in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # CE loss only on K known classes (labels 0..K-1 index into K+1 outputs)
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            self.optimizer_g.zero_grad()
            self.optimizer_c.zero_grad()
            src_cls_loss.backward()
            self.optimizer_g.step()
            self.optimizer_c.step()

            avg_meter['Pr_Src_cls_loss'].update(src_cls_loss.item(), src_x.size(0))

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        self.train()

        joint_loader = zip(src_loader, itertools.cycle(trg_loader))
        for (src_x, src_y), (trg_x, *_) in joint_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)
            trg_x = trg_x.to(self.device)

            # --- Step 1: Source classification loss ---
            self.optimizer_g.zero_grad()
            self.optimizer_c.zero_grad()

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)  # (B, K+1)
            loss_s = self.cross_entropy(src_pred, src_y)
            loss_s.backward()

            # --- Step 2: Target adversarial loss with GRL ---
            lambd = self.hparams.get('lambda_adv', 1.0)
            self.classifier.set_lambda(lambd)

            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat, reverse=True)  # GRL applied
            trg_prob = F.softmax(trg_pred, dim=1)

            # Collapse K+1 outputs to binary: (sum of K known, unknown)
            prob_known = trg_prob[:, :self.num_classes].sum(dim=1, keepdim=True)
            prob_unknown = trg_prob[:, self.num_classes].unsqueeze(1)
            prob_binary = torch.cat([prob_known, prob_unknown], dim=1)  # (B, 2)

            # BCE against uniform (0.5, 0.5) target
            target_uniform = torch.full_like(prob_binary, 0.5)
            loss_t = self._bce_loss(prob_binary, target_uniform)
            loss_t.backward()

            # --- Step 3: Single optimizer step (combined gradients) ---
            self.optimizer_g.step()
            self.optimizer_c.step()

            losses = {
                'Src_cls_loss': loss_s.item(),
                'Trg_adv_loss': loss_t.item(),
                'Total_loss': (loss_s + loss_t).item(),
            }
            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

    @staticmethod
    def _bce_loss(output, target):
        """Binary cross-entropy on probability vectors (not logits)."""
        output_neg = 1.0 - output
        target_neg = 1.0 - target
        result = torch.mean(target * torch.log(output + 1e-6))
        result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
        return -result

    def evaluate(self, test_loader, trg_private_class, src=False):
        self.feature_extractor.eval()
        self.classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels, *_ in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                features = self.feature_extractor(data)
                predictions = self.classifier(features)  # (B, K+1)

                # For loss computation, mask out private classes
                mask = labels < self.num_classes
                if mask.any():
                    loss = F.cross_entropy(predictions[mask], labels[mask])
                else:
                    loss = torch.tensor(0.0)
                total_loss.append(loss.detach().cpu().item())

                preds_list.append(predictions.detach())
                labels_list.append(labels)

        loss = torch.tensor(total_loss).mean()
        full_preds = torch.cat(preds_list)
        full_labels = torch.cat(labels_list)
        return loss, full_preds, full_labels

    def decision_function(self, preds):
        """Native K+1 decision: if argmax is the unknown class (K), return -1."""
        probs = F.softmax(preds, dim=1)
        pred_class = probs.argmax(dim=1)
        # Class K (the K+1th) is the unknown class
        pred_class[pred_class == self.num_classes] = -1
        return pred_class


class TSFA(Algorithm):
    """
    Two-Stage Feature Alignment (TNNLS 2026).

    Universal Open-Set Domain Adaptation for Time-Series via SDE and
    Optimal Transport.

    Stage 1 (Global): SDE-based alignment maps source and target features
        into a shared latent space by matching distribution moments.
    Stage 2 (Local): OSAM (Open-Set Alignment Module) uses class-aware
        optimal transport with pseudo-labels for fine-grained alignment.
        Samples with transport mass < 1/C^s are rejected as unknown.

    Key differences from existing methods:
        - Time-frequency features via STFT (not just time-domain CNN or global FFT)
        - SDE for global alignment (stochastic process, not adversarial)
        - OSAM for class-aware local alignment with built-in unknown rejection
    """
    SCENARIO = "OSDA"

    def __init__(self, backbone, configs, hparams, device):
        # Skip Algorithm.__init__ — TSFA uses its own TF feature extractor
        torch.nn.Module.__init__(self)
        self.configs = configs
        self.hparams = hparams
        self.device = device
        self.cross_entropy = nn.CrossEntropyLoss()
        self.num_classes = configs.num_classes

        # Time-frequency feature extractor
        self.feature_extractor = TFFeatureExtractor(backbone, configs).to(self.device)
        feat_dim = self.feature_extractor.out_dim

        # Neural SDE for global alignment
        sde_hidden = hparams.get('sde_hidden_dim', 128)
        sde_steps = hparams.get('sde_num_steps', 10)
        self.sde = NeuralSDE(feat_dim, hidden_dim=sde_hidden, num_steps=sde_steps).to(self.device)

        # Classifier on SDE output
        self.classifier = nn.Linear(feat_dim, configs.num_classes).to(self.device)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # OSAM threshold: 1/C^s (paper Eq. 11)
        self.osam_threshold = 1.0 / configs.num_classes

        # Optimizers
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.sde.parameters()) +
            list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
        )

        # For storing pseudo-labels and transport plan from OSAM
        self.transport_masses = None

    def update(self, src_loader, trg_loader, avg_meter, val_loader, logger):
        best_src_risk = float('inf')
        best_model = None
        best_model_epoch = None

        # Phase 1: Pretrain source classifier
        nb_pr_epochs = self.hparams.get("num_epochs_pr", 0)
        for epoch in range(1, nb_pr_epochs + 1):
            self._pretrain_epoch(src_loader, avg_meter)
            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        # Phase 2: Two-stage alignment training
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.state_dict())
                best_model_epoch = epoch

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = deepcopy(self.state_dict())
        if best_model is None:
            best_model = deepcopy(last_model)
            print(f"[TSFA-debug] best_model NEVER updated -> falling back to last (epoch {self.hparams['num_epochs']})")
        else:
            print(f"[TSFA-debug] best_model captured at epoch {best_model_epoch} (src_cls_loss={best_src_risk:.4f})")
        # Final per-loss state for diagnostic comparison across runs
        diag = {k: f"{v.avg:.4f}" for k, v in avg_meter.items()
                if k in ('Src_cls_loss', 'SDE_loss', 'OSAM_loss', 'Total_loss')}
        print(f"[TSFA-debug] final-epoch losses: {diag}")
        # Roll back to best (validation-selected) state for downstream eval.
        # last_model can be NaN-poisoned from late-epoch SDE divergence; the
        # framework runs calculate_metrics() against self.algorithm, so we
        # load best_model here. last_model is still returned for checkpointing.
        self.load_state_dict(best_model)
        print(f"[TSFA-debug] loaded best_model into algorithm for eval")
        return last_model, best_model

    def _pretrain_epoch(self, src_loader, avg_meter):
        """Source-only pretraining with TF features + SDE."""
        self.train()
        for src_x, src_y in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            # TF features -> SDE -> classifier
            feat = self.feature_extractor(src_x)
            z = self.sde(feat)
            pred = self.classifier(z)

            loss = self.cross_entropy(pred, src_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_meter['Pr_Src_cls_loss'].update(loss.item(), src_x.size(0))

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        self.train()

        joint_loader = zip(src_loader, itertools.cycle(trg_loader))
        for (src_x, src_y), (trg_x, *_) in joint_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)
            trg_x = trg_x.to(self.device)

            self.optimizer.zero_grad()

            # Extract time-frequency features
            feat_s = self.feature_extractor(src_x)
            feat_t = self.feature_extractor(trg_x)

            # === Stage 1: SDE Global Alignment ===
            sde_loss, z_s, z_t = self.sde.kl_divergence_loss(feat_s, feat_t)

            # Source classification on SDE output
            pred_s = self.classifier(z_s)
            cls_loss = self.cross_entropy(pred_s, src_y)

            # === Stage 2: OSAM Local Alignment ===
            osam_loss = self._osam_loss(z_s, src_y, z_t)

            # Combined loss
            lambda_sde = self.hparams.get('lambda_sde', 1.0)
            lambda_osam = self.hparams.get('lambda_osam', 1.0)
            loss = cls_loss + lambda_sde * sde_loss + lambda_osam * osam_loss

            loss.backward()
            self.optimizer.step()

            losses = {
                'Src_cls_loss': cls_loss.item(),
                'SDE_loss': sde_loss.item(),
                'OSAM_loss': osam_loss.item(),
                'Total_loss': loss.item(),
            }
            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

    def _osam_loss(self, z_s, y_s, z_t):
        """Open-Set Alignment Module: class-aware OT with pseudo-labels.

        For each source class c:
          - Select source samples of class c
          - Compute cost matrix between class-c source and all target
          - Solve OT to get transport plan
          - Weight alignment by transport mass (high mass = likely same class)

        Samples with total received mass < 1/C^s are considered unknown.
        """
        Ns, Nt = z_s.shape[0], z_t.shape[0]
        if Ns == 0 or Nt == 0:
            return torch.tensor(0.0, device=self.device)

        # Get pseudo-labels for target via current classifier
        with torch.no_grad():
            pred_t = self.classifier(z_t)
            pseudo_labels = pred_t.argmax(dim=1)
            confidence = F.softmax(pred_t, dim=1).max(dim=1).values

        total_loss = torch.tensor(0.0, device=self.device)
        n_classes_seen = 0

        for c in range(self.num_classes):
            # Source samples of class c
            src_mask = y_s == c
            if src_mask.sum() < 2:
                continue

            # Target samples pseudo-labeled as class c with confidence > threshold
            trg_mask = (pseudo_labels == c) & (confidence > self.osam_threshold)
            if trg_mask.sum() < 2:
                continue

            z_s_c = z_s[src_mask]
            z_t_c = z_t[trg_mask]

            # Cost matrix: pairwise L2 distance
            C = torch.cdist(z_s_c, z_t_c, p=2)

            # Solve OT with POT library
            with torch.no_grad():
                a = ot.unif(z_s_c.shape[0])
                b = ot.unif(z_t_c.shape[0])
                gamma = ot.emd(a, b, C.detach().cpu().numpy())
                gamma = torch.from_numpy(gamma).float().to(self.device)

            # Wasserstein loss: sum of transport_plan * cost
            ot_loss = (gamma * C).sum()
            total_loss = total_loss + ot_loss
            n_classes_seen += 1

        if n_classes_seen > 0:
            total_loss = total_loss / n_classes_seen

        return total_loss

    def evaluate(self, test_loader, trg_private_class, src=False):
        self.feature_extractor.eval()
        self.sde.eval()
        self.classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels, *_ in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # TF features -> SDE (deterministic at eval: no noise)
                feat = self.feature_extractor(data)
                z = self._sde_deterministic(feat)
                predictions = self.classifier(z)

                mask = labels < predictions.shape[-1]
                if mask.any():
                    loss = F.cross_entropy(predictions[mask], labels[mask])
                else:
                    loss = torch.tensor(0.0)
                total_loss.append(loss.detach().cpu().item())

                preds_list.append(predictions.detach())
                labels_list.append(labels)

        loss = torch.tensor(total_loss).mean()
        full_preds = torch.cat(preds_list)
        full_labels = torch.cat(labels_list)

        # For target: apply OSAM-based unknown detection
        if not src:
            full_preds = self._osam_reject(full_preds)

        return loss, full_preds, full_labels

    def _sde_deterministic(self, z):
        """Run SDE forward pass without stochastic noise (for evaluation)."""
        B = z.shape[0]
        for step in range(self.sde.num_steps):
            t = torch.full((B, 1), step * self.sde.dt, device=z.device)
            zt = torch.cat([z, t], dim=1)
            drift = self.sde.drift(zt)
            z = z + drift * self.sde.dt
        return z

    def _osam_reject(self, predictions):
        """Reject unknown samples using partial OT transport mass.

        Uses partial OT between source class centroids and target softmax
        probabilities. Only a fraction of target mass gets transported —
        samples receiving little mass are likely unknown.

        The mass ratio `m` is set to num_classes / (num_classes + 1), assuming
        roughly 1 unknown "class" worth of mass should be left unmatched.
        Target samples with received mass < 1/C^s are rejected.
        """
        Nt = predictions.shape[0]
        if Nt == 0:
            return predictions

        probs = F.softmax(predictions, dim=1)

        # Source centroids: ideal one-hot per known class
        src_centroids = torch.eye(self.num_classes, device=predictions.device)
        C = torch.cdist(src_centroids, probs, p=2)  # (K, Nt)

        with torch.no_grad():
            a = ot.unif(self.num_classes)
            b = ot.unif(Nt)
            # Partial OT: only transport fraction m of total mass
            m = self.num_classes / (self.num_classes + 1.0)
            C_np = C.cpu().numpy()
            # POT's partial_wasserstein hits "increase the number of dummy points"
            # when the LP solver runs out of slack — retry with more dummies, then
            # fall back to softmax-confidence rejection if it still fails.
            gamma = None
            used_nb_dummies = None
            for nb_dummies in (1, 10, max(50, Nt // 10)):
                try:
                    gamma = ot.partial.partial_wasserstein(a, b, C_np, m=m, nb_dummies=nb_dummies)
                    used_nb_dummies = nb_dummies
                    break
                except ValueError:
                    continue
            if gamma is None:
                # Confidence fallback: low-max-softmax samples treated as unknown.
                max_probs, _ = probs.max(dim=1)
                unknown_mask = max_probs < (1.0 / self.num_classes)
                n_unk = int(unknown_mask.sum().item())
                print(f"[TSFA-debug] OSAM-reject FALLBACK (softmax-conf): Nt={Nt} "
                      f"max_probs mean={max_probs.mean():.4f} std={max_probs.std():.4f} "
                      f"unknown={n_unk} ({n_unk/Nt:.1%})")
                predictions[unknown_mask] *= 0
                return predictions
            gamma = torch.from_numpy(gamma).float().to(predictions.device)

        # Total mass received by each target sample (normalized)
        target_mass = gamma.sum(dim=0)
        # Samples receiving less than threshold mass -> unknown
        mass_threshold = target_mass.mean() * 0.5  # adaptive: half the average mass
        unknown_mask = target_mass < mass_threshold
        n_unk = int(unknown_mask.sum().item())
        print(f"[TSFA-debug] OSAM-reject OT-OK: Nt={Nt} nb_dummies={used_nb_dummies} "
              f"mass mean={target_mass.mean():.4e} std={target_mass.std():.4e} "
              f"thr={mass_threshold:.4e} unknown={n_unk} ({n_unk/Nt:.1%})")
        predictions[unknown_mask] *= 0
        return predictions

    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred


class SPADA(Algorithm):
    """
    Stacked auto-encoder based Partial Adversarial Domain Adaptation
    (Liu et al., IEEE TII 2021).

    Partial DA: target classes are a subset of source classes. Two domain
    discriminators are used:
      - Dw (weighting): scores each sample as source-vs-target. Source samples
        whose Dw output is close to 1 are likely outlier (source-private)
        classes; their weight mu_tilde = 1 - sigmoid(Dw(f_s)) becomes small,
        suppressing negative transfer.
      - Dd (adversarial): standard GRL-based domain discriminator that aligns
        the weighted source distribution with the target distribution.
    A label predictor (the standard classifier) is trained on weighted source
    cross-entropy.
    """
    SCENARIO = "PDA"

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)
        self.hparams = hparams
        self.device = device
        self.num_classes = configs.num_classes

        feat_dim = configs.feat_dim
        hid = getattr(configs, 'disc_hid_dim', 256)

        def _disc():
            return nn.Sequential(
                nn.Linear(feat_dim, hid),
                nn.ReLU(),
                nn.Linear(hid, hid),
                nn.ReLU(),
                nn.Linear(hid, 1),
            )

        self.Dw = _disc()  # weighting discriminator (no GRL)
        self.Dd = _disc()  # adversarial discriminator (with GRL)

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.Dd.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
        )
        self.optimizer_dw = torch.optim.Adam(
            self.Dw.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
        )
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.bce_logits_none = nn.BCEWithLogitsLoss(reduction='none')

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, *_)) in joint_loader:
            src_x = src_x.to(self.device)
            src_y = src_y.to(self.device)
            trg_x = trg_x.to(self.device)

            p = float(step + epoch * num_batches) / (self.hparams["num_epochs"] * num_batches + 1)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            src_feat = self.feature_extractor(src_x)
            trg_feat = self.feature_extractor(trg_x)

            src_flat = src_feat.reshape(src_feat.size(0), -1)
            trg_flat = trg_feat.reshape(trg_feat.size(0), -1)

            # ---- Step 1: train Dw on detached features ----
            dw_src = self.Dw(src_flat.detach())
            dw_trg = self.Dw(trg_flat.detach())
            dw_loss = 0.5 * (
                self.bce_logits(dw_src, torch.ones_like(dw_src)) +
                self.bce_logits(dw_trg, torch.zeros_like(dw_trg))
            )
            self.optimizer_dw.zero_grad()
            dw_loss.backward()
            self.optimizer_dw.step()

            # ---- Compute source weights mu_tilde = 1 - sigma(Dw(f_s)) ----
            with torch.no_grad():
                w_src = 1.0 - torch.sigmoid(self.Dw(src_flat)).squeeze(1)
                w_src = w_src / (w_src.mean() + 1e-8)  # normalize batch mean to 1

            # ---- Step 2: weighted CE + GRL adversarial loss ----
            src_pred = self.classifier(src_feat)
            src_cls_loss = (F.cross_entropy(src_pred, src_y, reduction='none') * w_src).mean()

            src_rev = ReverseLayerF.apply(src_flat, alpha)
            trg_rev = ReverseLayerF.apply(trg_flat, alpha)
            dd_src = self.Dd(src_rev)
            dd_trg = self.Dd(trg_rev)

            src_dom_loss = (
                self.bce_logits_none(dd_src, torch.ones_like(dd_src)).squeeze(1) * w_src
            ).mean()
            trg_dom_loss = self.bce_logits(dd_trg, torch.zeros_like(dd_trg))
            adv_loss = src_dom_loss + trg_dom_loss

            loss = self.hparams.get("src_cls_loss_wt", 1.0) * src_cls_loss + \
                   self.hparams.get("domain_loss_wt", 1.0) * adv_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {
                'Total_loss': loss.item(),
                'Src_cls_loss': src_cls_loss.item(),
                'Adv_loss': adv_loss.item(),
                'Dw_loss': dw_loss.item(),
            }
            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))


class PDAAN(Algorithm):
    """
    Partial Domain Adaptation Adversarial Network (Zhou et al., MST 2022).

    Partial DA: target classes are a subset of source classes. Three losses:
      - L_cls^w (eq. 2): class-weighted source CE, where each sample is scaled
        by omega[y] -- the running average of target softmax predictions per
        class. Source-private classes get small omega and contribute little.
      - L_wce^w (eqs. 3-4): class-weighted complement entropy regularizer
        encouraging high entropy over non-target classes (focuses prediction
        mass on the true class).
      - L_adv^E (eqs. 6-7): GRL adversarial loss against a domain
        discriminator, with two extras: (i) per-sample entropy weight
        m(x) = 1 + exp(-H(D(G(x)))) emphasising hard-to-align samples, and
        (ii) a "domain expansion" term that, with weight rho decaying to 0,
        treats some source samples as target to bootstrap alignment.
    """
    SCENARIO = "PDA"

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)
        self.hparams = hparams
        self.device = device
        self.num_classes = configs.num_classes

        self.domain_classifier = DiscriminatorUDA(configs)

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.domain_classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
        )

        # Running estimate of class-level weights omega (eq. 1).
        self.register_buffer('class_weights', torch.ones(self.num_classes))
        self.cw_momentum = hparams.get('class_weights_momentum', 0.9)

    def _complement_entropy(self, softmax_pred, labels, xi):
        """l_wce per sample (eq. 4)."""
        eps = 1e-7
        y_a = softmax_pred.gather(1, labels.unsqueeze(1)).squeeze(1)
        denom = (1 - y_a).clamp(min=eps).unsqueeze(1)
        renorm = softmax_pred / denom
        mask = torch.ones_like(renorm)
        mask.scatter_(1, labels.unsqueeze(1), 0)
        renorm = renorm * mask
        log_renorm = torch.log(renorm.clamp(min=eps)) * mask
        ent_term = (renorm * log_renorm).sum(dim=1)
        return (1 - y_a).pow(xi) * ent_term

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        num_epochs = self.hparams["num_epochs"]

        alpha_w = self.hparams.get("alpha", 1.0)
        beta_w = self.hparams.get("beta", 1.0)
        xi = self.hparams.get("xi", 1.0)
        rho_init = self.hparams.get("rho_init", 1.0)
        # rho decays linearly to 0 across training (paper sec. 3.2.3).
        rho = max(0.0, rho_init * (1.0 - (epoch - 1) / max(num_epochs - 1, 1)))
        eps = 1e-7

        for step, ((src_x, src_y), (trg_x, *_)) in joint_loader:
            src_x = src_x.float().to(self.device)
            src_y = src_y.long().to(self.device)
            trg_x = trg_x.float().to(self.device)

            p = float(step + (epoch - 1) * num_batches) / (num_epochs * num_batches + 1)
            grl_alpha = 2. / (1. + np.exp(-10 * p)) - 1

            src_feat = self.feature_extractor(src_x)
            trg_feat = self.feature_extractor(trg_x)
            src_flat = src_feat.reshape(src_feat.size(0), -1)
            trg_flat = trg_feat.reshape(trg_feat.size(0), -1)

            src_logits = self.classifier(src_feat)
            trg_logits = self.classifier(trg_feat)
            trg_softmax = F.softmax(trg_logits, dim=1)
            src_softmax = F.softmax(src_logits, dim=1)

            # ---- Update class-level weights omega from target predictions (eq. 1) ----
            with torch.no_grad():
                batch_omega = trg_softmax.mean(dim=0)
                self.class_weights.mul_(self.cw_momentum).add_(
                    batch_omega, alpha=(1 - self.cw_momentum)
                )
                omega_norm = self.class_weights / (self.class_weights.max() + eps)
            w_per_src = omega_norm[src_y]

            # ---- Loss 1: class-weighted classification loss (eq. 2) ----
            ce_per_sample = F.cross_entropy(src_logits, src_y, reduction='none')
            cls_loss = (w_per_src * ce_per_sample).mean()

            # ---- Loss 2: class-weighted complement entropy loss (eqs. 3-4) ----
            wce_per_sample = self._complement_entropy(src_softmax, src_y, xi)
            denom_log = max(np.log(max(self.num_classes - 1, 2)), eps)
            wce_loss = (w_per_src * wce_per_sample).mean() / denom_log

            # ---- Loss 3: domain enhancement adversarial loss (eqs. 5-7) ----
            src_rev = ReverseLayerF.apply(src_flat, grl_alpha)
            trg_rev = ReverseLayerF.apply(trg_flat, grl_alpha)
            d_src = self.domain_classifier(src_rev).squeeze(1).clamp(eps, 1 - eps)
            d_trg = self.domain_classifier(trg_rev).squeeze(1).clamp(eps, 1 - eps)

            with torch.no_grad():
                H_src = -(d_src * torch.log(d_src) + (1 - d_src) * torch.log(1 - d_src))
                H_trg = -(d_trg * torch.log(d_trg) + (1 - d_trg) * torch.log(1 - d_trg))
                m_src = 1.0 + torch.exp(-H_src)
                m_trg = 1.0 + torch.exp(-H_trg)

            # Standard GRL adversarial: source label 1, target label 0.
            adv_h_src = -(m_src * torch.log(d_src)).mean()
            adv_h_trg = -(m_trg * torch.log(1 - d_trg)).mean()
            adv_h_loss = adv_h_src + adv_h_trg

            # Domain expansion term: a portion of source samples is treated as
            # target (eq. 6, second sum), weighted by m, omega and rho.
            adv_exp_loss = -rho * (m_src * w_per_src * torch.log(1 - d_src)).mean()

            adv_loss = adv_h_loss + adv_exp_loss

            # ---- Total (eq. 8) ----
            loss = cls_loss + alpha_w * wce_loss + beta_w * adv_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {
                'Total_loss': loss.item(),
                'Src_cls_loss': cls_loss.item(),
                'WCE_loss': wce_loss.item(),
                'Adv_loss': adv_loss.item(),
            }
            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))
