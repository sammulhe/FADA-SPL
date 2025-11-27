import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed():
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 梯度反转层，把在domain classifier中传过来的导数
class reverse_layer(Function):
    """
    自定义计算方式
    """

    @staticmethod
    def forward(self, x, alpha):
        self.alpha = alpha
        return x

    @staticmethod
    def backward(self, grad_output):
        output = grad_output.neg() * self.alpha
        return output, None


"""
domain classifier
"""


class Domain_classifier(nn.Module):
    def __init__(self, features, units, dropout=0.2):
        super(Domain_classifier, self).__init__()
        classifier = nn.Sequential(
            nn.Linear(128, 500, bias=False),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(500, 500, bias=False),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(500, 2)
        )
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(x)


"""
task classifier
"""


class Task_classifier(nn.Module):
    def __init__(self, features, units, classes, dropout=0.2):
        super(Task_classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(128, classes)
        )

    def forward(self, x):
        return self.classifier(x)


"""
attention lstm feature extract
"""


class cnn_feature_extract(nn.Module):
    def __init__(self, features):
        super(cnn_feature_extract, self).__init__()
        self.cov1 = nn.Conv1d(features, 128, 8, stride=1, padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

        self.cov2 = nn.Conv1d(128, 256, 5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(256)

        self.cov3 = nn.Conv1d(256, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # self.pool = nn.AvgPool1d(2)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # self.nn = nn.Linear(32000, num_classes)

    def forward(self, x):  # 输入为NxCxL, N是batch size， C是channel（即时间序列的features）, L是length
        # n_samples = x.size(0)
        ############第一个卷积层###########################
        out = self.cov1(x)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.relu(out)
        ############第一个卷积层###########################

        out = self.cov2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # print(out.shape)
        ##############################################
        out = self.cov3(out)
        out = self.bn3(out)
        out = self.relu(out)
        # print(out.shape)

        ############pooling层#############
        out = self.pool(out).squeeze(-1)
        #########################

        return out

class alstm_feature_extract(nn.Module):
    def __init__(self, features, units=128, layers=1):
        super(alstm_feature_extract, self).__init__()
        # self.feature_map = nn.Linear(features, features)
        self.feature_map = nn.Sequential(
            nn.Linear(features, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(units, units, layers, batch_first=True)
        self.norm = nn.BatchNorm1d(units)
        self.softmax = nn.Softmax(dim=1)
        self.a_W = nn.init.uniform_(
            torch.empty((units, units), dtype=torch.float32, device=device, requires_grad=True),
            a=0, b=1)
        self.a_b = torch.zeros(units, dtype=torch.float32, device=device, requires_grad=True)
        self.a_u = nn.init.uniform_(torch.empty(units, dtype=torch.float32, device=device, requires_grad=True), a=0,
                                    b=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feature_map(x)
        outputs, _ = self.lstm(x)  ###bacth*timestep*feature

        ###src和trg共用W, b, u 的attention计算weights##########
        attn_latent = torch.tanh(torch.tensordot(outputs, self.a_W, dims=1) + self.a_b)
        a_scores = torch.tensordot(attn_latent, self.a_u, dims=1)
        a_weights = self.softmax(a_scores)
        ##############################################

        #############combined hidden state#########################
        combined = torch.sum(outputs * a_weights.unsqueeze(-1), 1)

        ###########send the hidden state to the linear layer###########
        combined = self.norm(combined)

        return combined



class codats_model(nn.Module):
    def __init__(self, features, rnn_units, classifier_units, layers, classes, dropout=0.2):
        #set_seed()
        super(codats_model, self).__init__()
        self.feature_extract = cnn_feature_extract(features)
        self.task_classifier = Task_classifier(rnn_units, classifier_units, classes, dropout)
        self.domain_classifier = Domain_classifier(rnn_units, classifier_units, dropout)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, source, target, alpha=1):
        src_ext = self.feature_extract(source)
        trg_ext = self.feature_extract(target)
        src_pred = self.task_classifier(src_ext)
        trg_pred = self.task_classifier(trg_ext)

        #pdist = nn.PairwiseDistance(p=2)
        similarity = self.cos(trg_ext.unsqueeze(-1),
                              src_ext.unsqueeze(0).transpose(1, 2))  # batch*feature*1, 1*features*batch
        similarity = similarity * 10
        #nn.functional.one_hot(src_Y, num_classes=num_class).to(dtype=src_pred.dtype)

        #max_sim, index = similarity.topk(1, dim=1)

        fea_ext = torch.cat((src_ext, trg_ext), dim=0)
        fea_ext = reverse_layer.apply(fea_ext, alpha)

        src_domain_label = torch.ones(len(source), device=device).long()  # 来自source的就是1
        trg_domain_label = torch.zeros(len(target), device=device).long()  # 来自target的就是0
        domain_label = torch.cat((src_domain_label, trg_domain_label), dim=0)

        domain_pred = self.domain_classifier(fea_ext)

        return src_pred, trg_pred, domain_pred, domain_label, trg_ext, similarity

    def predict(self, x):
        out = self.feature_extract(x)
        out = self.task_classifier(out)
        return out

    def get_hidden(self, source, target):
        src_fea_ext = self.feature_extract(source)
        trg_fea_ext = self.feature_extract(target)
        return src_fea_ext, trg_fea_ext


# class codats_model(nn.Module):
#     def __init__(self, features, rnn_units, classifier_units, layers, classes, dropout=0.2):
#         set_seed()
#         super(codats_model, self).__init__()
#         self.feature_extract = cnn_feature_extract(features)
#         self.task_classifier = Task_classifier(rnn_units, classifier_units, classes, dropout)
#         self.domain_classifier = Domain_classifier(rnn_units, classifier_units, dropout)
#
#     def forward(self, x, source=True, alpha=1):
#         fea_ext = self.feature_extract(x)
#         task_pred = self.task_classifier(fea_ext)
#
#         fea_ext = reverse_layer.apply(fea_ext, alpha)
#         if source:
#             domain_label = torch.ones(len(x), device=device).long()  # 来自source的就是1
#         else:
#             domain_label = torch.zeros(len(x), device=device).long()  # 来自target的就是0
#         domain_pred = self.domain_classifier(fea_ext)
#
#         return task_pred, domain_pred, domain_label
#
#     def get_hidden(self, source, target):
#         src_fea_ext = self.feature_extract(source)
#         trg_fea_ext = self.feature_extract(target)
#         return src_fea_ext, trg_fea_ext


class SpectralMap(nn.Module):
    def __init__(self, features, units):
        super(SpectralMap, self).__init__()

        self.layer = nn.Linear(features, units)  # klayer.Dense(self.units,kernel_initializer='random_uniform')
        nn.init.uniform_(self.layer.weight)
        self.norm = nn.BatchNorm1d(units)
        # self.act = torch.cos()#klayer.Activation(activation=CosActivation)

    def forward(self, x):
        layer_out = self.layer(x)
        out_norm = self.norm(layer_out)
        # print(torch.cos(out_norm).shape)
        return torch.cos(out_norm)


class SpectralMap2(nn.Module):
    def __init__(self, features, units):
        super(SpectralMap2, self).__init__()
        self.layer1 = nn.Linear(features, int(units / 2))
        self.layer2 = nn.Linear(features, int(units / 2))
        self.norm = nn.BatchNorm1d(int(units / 2))
        self.act2 = nn.ReLU()

    def forward(self, x):
        layer_out1 = self.layer1(x)
        layer_out2 = self.layer2(x)
        out_norm1 = self.norm(layer_out1)
        out_norm2 = self.norm(layer_out2)
        out1 = torch.cos(out_norm1)
        out2 = self.act2(out_norm2)
        return torch.cat([out1, out2], axis=1)


class DSKN(nn.Module):
    def __init__(self, features):
        super(DSKN, self).__init__()
        self.layer1 = SpectralMap(features, 256)
        self.layer2 = SpectralMap(256, 128)

    def forward(self, x):
        layer_out1 = self.layer1(x)
        layer_out2 = self.layer2(layer_out1)
        return layer_out2


class DSKN2(nn.Module):
    def __init__(self, features):
        super(DSKN2, self).__init__()
        self.layer1 = SpectralMap2(features, 256)
        self.layer2 = SpectralMap2(256, 128)

    def forward(self, x):
        layer_out1 = self.layer1(x)
        layer_out2 = self.layer2(layer_out1)
        return layer_out2


class askmda_model(nn.Module):
    def __init__(self, features, rnn_units, classifier_units, layers, classes, dropout=0.2):
        #set_seed()
        super(askmda_model, self).__init__()
        self.feature_extract = cnn_feature_extract(features)
        self.domain_embeded = DSKN2(features)
        self.task_classifier = Task_classifier(rnn_units, classifier_units, classes, dropout)

    def forward(self, source, target):
        source_out = self.feature_extract(source)
        target_out = self.feature_extract(target)

        source_out_emd = self.domain_embeded(source_out)
        target_out_emd = self.domain_embeded(target_out)

        task_pred = self.task_classifier(source_out)

        delta = torch.mean(source_out_emd, axis=0) - torch.mean(target_out_emd, axis=0)
        domain_loss = torch.sum(torch.mul(delta, delta))

        return task_pred, domain_loss
