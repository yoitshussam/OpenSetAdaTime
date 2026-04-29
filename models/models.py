import torch
from torch import nn
import math
from torch.autograd import Function
from torch.nn.utils import weight_norm
import torch.nn.functional as F



def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


##################################################
##########  BACKBONE NETWORKS  ###################
##################################################

########## CNN #############################

class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.cnn_blocks[0]['kernel_size'],
                      stride=configs.stride, bias=False, padding=(configs.cnn_blocks[0]['kernel_size'] // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
        )
        if configs.cnn_blocks[0]['maxpool']:
            self.conv_block1.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        if configs.cnn_blocks[0]['dropout']:
            self.conv_block1.append(nn.Dropout(configs.dropout))

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=configs.cnn_blocks[1]['kernel_size'], stride=1, bias=False, padding=(configs.cnn_blocks[1]['kernel_size'] // 2)),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
        )
        if configs.cnn_blocks[1]['maxpool']:
            self.conv_block2.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        if configs.cnn_blocks[1]['dropout']:
            self.conv_block2.append(nn.Dropout(configs.dropout))

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=configs.cnn_blocks[2]['kernel_size'], stride=1, bias=False,
                      padding=(configs.cnn_blocks[2]['kernel_size'] // 2)),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
        )
        if configs.cnn_blocks[2]['maxpool']:
            self.conv_block3.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        if configs.cnn_blocks[2]['dropout']:
            self.conv_block3.append(nn.Dropout(configs.dropout))

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


########## FNO (CNN + Fourier) #############################

class SpectralConv1d(nn.Module):
    """1D Fourier layer: FFT -> linear transform in spectral domain -> inverse FFT."""
    def __init__(self, in_channels, out_channels, modes1, fl=128):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x = torch.cos(x)
        x_ft = torch.fft.rfft(x, norm='ortho')
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        r = out_ft[:, :, :self.modes1].abs()
        p = out_ft[:, :, :self.modes1].angle()
        return torch.concat([r, p], -1), out_ft


class FNO(nn.Module):
    """CNN + Fourier Neural Operator backbone (from UniDABench).

    Concatenates time-domain CNN features with frequency-domain (FFT amplitude
    + phase) features. Output dim = CNN_out + 2*fourier_modes.
    """
    def __init__(self, configs):
        super(FNO, self).__init__()
        self.modes1 = configs.fourier_modes
        self.width = configs.input_channels
        self.length = configs.sequence_len
        self.freq_feature = SpectralConv1d(self.width, self.width, self.modes1, self.length)
        self.bn_freq = nn.BatchNorm1d(configs.fourier_modes * 2)
        self.cnn = CNN(configs)
        self.avg = nn.Conv1d(self.width, 1, kernel_size=3,
                             stride=configs.stride, bias=False, padding=(3 // 2))

    def forward(self, x):
        ef, out_ft = self.freq_feature(x)
        if ef.shape[0] == 1 and ef.shape[1] != 1:
            ef = self.avg(ef).squeeze(0)
        elif ef.shape[1] == 1:
            ef = ef.squeeze()
        else:
            ef = self.bn_freq(self.avg(ef).squeeze())
        ef = F.relu(ef)
        et = self.cnn(x)
        f = torch.concat([ef, et], -1)
        return F.normalize(f)


class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()
        self.logits = nn.Linear(configs.feat_dim, configs.num_classes)
        self.configs = configs

    def forward(self, x):
        predictions = self.logits(x)
        return predictions


##################################################
##########  OPEN-SET DA COMPONENTS  ##############
##################################################

class classifierOVANet(nn.Module):
    def __init__(self, configs):
        super(classifierOVANet, self).__init__()
        self.logits = nn.Linear(configs.feat_dim, configs.num_classes * 2)
        self.configs = configs

    def forward(self, x):
        predictions = self.logits(x)
        return predictions


class classifierOSBP(nn.Module):
    """K+1 classifier for OSBP: bottleneck + single linear layer.

    Architecture (matching original paper):
        Bottleneck: Linear→BN→LeakyReLU→Dropout→Linear→BN→LeakyReLU→Dropout
        Classifier: Single linear layer (bottleneck_dim → K+1)

    The bottleneck gives the GRL a richer feature space. The classifier is a
    single linear layer so GRL directly influences the decision boundary.
    """
    def __init__(self, configs):
        super(classifierOSBP, self).__init__()
        feat_dim = configs.feat_dim
        bottleneck_dim = getattr(configs, 'osbp_bottleneck_dim', 256)

        self.bottleneck = nn.Sequential(
            nn.Linear(feat_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )
        # Single linear layer — matching original paper
        self.logits = nn.Linear(bottleneck_dim, configs.num_classes + 1)
        self.num_classes = configs.num_classes
        self.lambd = 1.0

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        x = x.reshape(x.shape[0], -1)
        x = self.bottleneck(x)
        if reverse:
            x = ReverseLayerF.apply(x, self.lambd)
        return self.logits(x)


class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.0):
        super(LinearAverage, self).__init__()
        self.nLem = outputSize
        self.momentum = momentum
        self.register_buffer('params', torch.tensor([T, momentum]))
        self.register_buffer('memory', torch.zeros(outputSize, inputSize))
        self.flag = 0
        self.T = T

    def forward(self, x, y):
        out = torch.mm(x, self.memory.t()) / self.T
        return out

    def update_weight(self, features, index):
        if not self.flag:
            weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
            weight_pos.mul_(0.0)
            weight_pos.add_(torch.mul(features.data, 1.0))
            w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(w_norm)
            self.memory.index_copy_(0, index, updated_weight)
            self.flag = 1
        else:
            weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
            weight_pos.mul_(self.momentum)
            weight_pos.add_(torch.mul(features.data, 1 - self.momentum))
            w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(w_norm)
            self.memory.index_copy_(0, index, updated_weight)
        self.memory = F.normalize(self.memory)

    def set_weight(self, features, index):
        self.memory.index_copy_(0, index, features)


class DiscriminatorUDA(nn.Module):
    """Discriminator model for UDA/UAN."""

    def __init__(self, configs):
        super(DiscriminatorUDA, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(configs.feat_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        out = self.layer(input)
        return out


class ProtoCLS(nn.Module):
    """
    Prototype-based classifier: L2-norm + fc layer (without bias)
    """
    def __init__(self, in_dim, out_dim, temp=0.05):
        super(ProtoCLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.tmp = temp
        self.weight_norm()

    def forward(self, x):
        x = F.normalize(x)
        x = self.fc(x) / self.tmp
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))


class CLS(nn.Module):
    """
    Classifier with projection head and prototype-based classifier.
    Returns (before_lincls_feat, after_lincls).
    """
    def __init__(self, configs, temp=0.05):
        super(CLS, self).__init__()
        feat_dim = configs.feat_dim
        self.projection_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, configs.final_out_channels))
        self.ProtoCLS = ProtoCLS(configs.final_out_channels, configs.num_classes, temp)

    def forward(self, x):
        before_lincls_feat = self.projection_head(x)
        after_lincls = self.ProtoCLS(before_lincls_feat)
        return before_lincls_feat, after_lincls


class MemoryQueue(nn.Module):
    def __init__(self, feat_dim, batchsize, n_batch, T=0.05, device='cuda'):
        super(MemoryQueue, self).__init__()
        self.feat_dim = feat_dim
        self.batchsize = batchsize
        self.T = T
        self.device = device

        # init memory queue
        self.queue_size = self.batchsize * n_batch
        self.register_buffer('mem_feat', torch.zeros(self.queue_size, feat_dim))
        self.register_buffer('mem_id', torch.zeros((self.queue_size), dtype=int))

        # write pointer
        self.next_write = 0

    def forward(self, x):
        out = torch.mm(x, self.mem_feat.t()) / self.T
        return out

    def get_nearest_neighbor(self, anchors, id_anchors=None):
        feat_mat = self.forward(anchors)

        if id_anchors is not None:
            A = id_anchors.reshape(-1, 1).repeat(1, self.mem_id.size(0))
            B = self.mem_id.reshape(1, -1).repeat(id_anchors.size(0), 1)
            mask = torch.eq(A, B)
            id_mask = torch.nonzero(mask)
            temp = id_mask[:, 1]
            feat_mat[:, temp] = -1 / self.T

        values, indices = torch.max(feat_mat, 1)
        nearest_feat = torch.zeros((anchors.size(0), self.feat_dim)).to(self.device)
        for i in range(anchors.size(0)):
            nearest_feat[i] = self.mem_feat[indices[i], :]
        return values, nearest_feat

    def update_queue(self, features, ids):
        w_ids = torch.arange(self.next_write, self.next_write + self.batchsize).to(self.device)
        self.mem_feat.index_copy_(0, w_ids, features.data)
        self.mem_id.index_copy_(0, w_ids, ids.data)
        self.mem_feat = F.normalize(self.mem_feat)

        self.next_write += self.batchsize
        if self.next_write == self.queue_size:
            self.next_write = 0

    def random_sample(self, size):
        id_t = torch.floor(torch.rand(size) * self.mem_feat.size(0)).long().to(self.device)
        sample_feat = self.mem_feat[id_t]
        return sample_feat


class ClassMemoryQueue(nn.Module):
    def __init__(self, feat_dim, num_classes, N, T=0.05):
        super(ClassMemoryQueue, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.N = N
        self.T = T

        self.register_buffer('mem_feat', torch.zeros(num_classes, N, feat_dim))
        self.register_buffer('mem_count', torch.zeros(num_classes, dtype=torch.int64))

    def forward(self, x):
        mem_feat_flat = self.mem_feat.view(-1, self.feat_dim)
        out = torch.mm(x, mem_feat_flat.t()) / self.T
        return out

    def update_queue(self, features, labels):
        with torch.no_grad():
            for i in range(self.num_classes):
                mask = labels == i
                if mask.any():
                    class_features = features[mask]
                    class_count = class_features.size(0)
                    current_count = self.mem_count[i].item()

                    if current_count + class_count <= self.N:
                        self.mem_feat[i, current_count:current_count + class_count] = class_features
                        self.mem_count[i] += class_count
                    else:
                        space_needed = self.N - min(self.N, class_count)
                        self.mem_feat[i, space_needed:] = class_features[:min(self.N, class_count)]
                        self.mem_count[i] = min(self.N, current_count + class_count)

    def get_class_features(self, class_idx):
        count = self.mem_count[class_idx].item()
        return self.mem_feat[class_idx, :count]

    def compute_distances(self, target_features):
        Nt = target_features.size(0)
        distances = torch.zeros(Nt, self.num_classes).to(target_features.device)

        for k in range(self.num_classes):
            class_features = self.get_class_features(k)
            if class_features.size(0) == 0:
                distances[:, k] = float('inf')
                continue
            dists = torch.cdist(target_features, class_features)
            distances[:, k] = dists.min(dim=1).values

        return distances

    def is_memory_full(self):
        return self.mem_count >= self.N
