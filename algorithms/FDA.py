"""
@author: Mingyang Liu
@contact: mingyang1024@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import ConditionalEntropyLoss
from algorithms.algorithms_base import Algorithm
from utils.module import *
from utils.build_model import *
from utils.transform import four_transform


class FDA(Algorithm):

    def __init__(self, configs, device, args):
        super(FDA, self).__init__(configs)

        # hyperparameters
        self.args = args
        self.device = device
        self.period = configs.period
        #self.avg_mode = configs.avg_mode
        self.fft_mode = self.period // 2 + 1
        #assert self.avg_mode < self.fft_mode
        self.kl_t = args.kl_t
        self.configs = configs

        # model
        self.feature_extractor = CNN(configs)
        self.task_classifier = TemporalClassifierHead(self.feature_extractor.out_dim, configs.num_classes)
        self.domain_classifier = Discriminator(self.feature_extractor.out_dim, self.args.disc_hid_dim)

        # self.feature_extractor = cnn_feature_extract(configs.input_channels)
        # self.task_classifier = Task_classifier(128, 128, configs.num_classes, 0.2)
        # self.domain_classifier = Domain_classifier(128, 128, 0.2)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # optimizers
        self.optimizer = torch.optim.Adam([
            {'params': self.feature_extractor.parameters()},
            {'params': self.task_classifier.parameters()},
            {'params': self.domain_classifier.parameters()}],
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        self.num_classes = configs.num_classes
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.kl = nn.KLDivLoss(reduction=args.kl_reduction)


    def get_amplitude(self, x_fft):
        a = x_fft.abs()
        if a.dim() == 4:
            a = a.mean(dim=2)
        a_disc = a[:, :, :self.fft_mode]
        a_disc = self.avg_pooling(a_disc.mean(dim=1)).softmax(-1)
        a_cls = a[:, :, :self.fft_mode]
        a_cls = a_cls.reshape(a_cls.size(0), -1)
        return a_cls, a_disc

    def update(self, src_x, src_y, trg_x):
        aug_src_x = four_transform(src_x, trg_x, self.configs.sequence_len, self.configs.input_channels, self.args.high_freq)

        # prepare true domain labels
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

        # source features and predictions
        src_t_feat = self.feature_extractor(aug_src_x)
        src_t_pred = self.task_classifier(src_t_feat)
        #aug_src_t_feat = self.feature_extractor(aug_src_x)
        #aug_src_t_pred = self.task_classifier(aug_src_t_feat)

        # target features and predictions
        trg_t_feat = self.feature_extractor(trg_x)
        trg_t_pred = self.task_classifier(trg_t_feat)


        similarity = self.cos(trg_t_feat.unsqueeze(-1),
                              src_t_feat.unsqueeze(0).transpose(1, 2))  # batch*feature*1, 1*features*batch
        similarity = similarity * 10
        #nn.functional.one_hot(src_Y, num_classes=num_class).to(dtype=src_pred.dtype)

        #max_sim, index = similarity.topk(1, dim=1)

        fea_ext = torch.cat((src_t_feat, trg_t_feat), dim=0)
        fea_ext = reverse_layer.apply(fea_ext, 1)
        disc_prediction = self.domain_classifier(fea_ext)
        domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)
        domain_acc = self.get_domain_acc(disc_prediction, domain_label_concat)


        new_src_Y = nn.functional.one_hot(src_y, num_classes=self.num_classes).t().to(self.device)
        sim_matrix = new_src_Y.unsqueeze(1).expand(new_src_Y.shape[0], similarity.shape[0], new_src_Y.shape[1]) * similarity
        max_sim, indices = torch.max(sim_matrix, dim=2)#sim_matrix.topk(1, dim=2)
        #print(max_sim)
        max_sim = max_sim.t()#max_sim.squeeze(2).t().to(device)
        prob = nn.Softmax(dim=1)(max_sim)
        #print('aa', prob)
        trg_t_pred = nn.LogSoftmax(dim=1)(trg_t_pred)

        task_loss = self.cross_entropy(src_t_pred, src_y)
        #aug_task_loss = self.cross_entropy(aug_src_t_pred, src_y)
        trg_loss = F.kl_div(trg_t_pred, prob,
                            reduction='batchmean')  # criterion(trg_pred, trg_pseudo)#F.kl_div(trg_pred, trg_pseudo, reduction='batchmean')
        loss = task_loss + domain_loss + self.args.entropy_trade_off * trg_loss# + aug_task_loss
           # update feature extractor
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Src_t_cls_loss': task_loss.item(),
                'Src_f_cls_loss': 0,
                'Domain_loss': domain_loss.item(),
                'align source tf loss': 0,
                'align target tf loss': 0,
                'cond_ent_loss_t': trg_loss.item(),
                'cond_ent_loss_f': 0,
                'domain acc': domain_acc.item()}

    '''return predictions'''

    def predict(self, data):
        self.feature_extractor.eval()
        self.task_classifier.eval()
        with torch.no_grad():
            t_feat = self.feature_extractor(data)
            pred = self.task_classifier(t_feat)
        return pred

    def save_model(self, path):
        torch.save({
            't_encoder': self.feature_extractor.state_dict(),
            't_classifier': self.task_classifier.state_dict(),
            'domain_classifier': self.domain_classifier.state_dict(),
            #'f_encoder': self.f_feature_extractor.state_dict(),
            #'f_classifier': self.f_classifier.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.feature_extractor.load_state_dict(checkpoint['t_encoder'])
        self.task_classifier.load_state_dict(checkpoint['t_classifier'])
        #self.f_feature_extractor.load_state_dict(checkpoint['f_encoder'])
        #self.f_classifier.load_state_dict(checkpoint['f_classifier'])

    def get_domain_acc(self, pred, label):
        pred = torch.argmax(pred, dim=1)
        res = torch.sum(torch.eq(pred, label)) / label.size(0)
        return res



