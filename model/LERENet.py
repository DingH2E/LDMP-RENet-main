import torch
from torch import nn
from torch.nn import BatchNorm2d as BatchNorm
import torch.nn.functional as F
import model.backbone.resnet as models
import model.backbone.vgg as vgg_models
from model.MPR import MPR
from model.MPE import MPE
import torch.nn.init as initer


def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):#, BatchNorm1d, BatchNorm2d, BatchNorm3d)):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)

class FSSNet(nn.Module):
    def __init__(self, layers=50, classes=2, criterion=nn.CrossEntropyLoss(ignore_index=255),
                 pretrained=True, shot=1, ppm_scales=[60, 30, 15, 8], vgg=False):
        super(FSSNet, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1

        # init parameters
        self.criterion = criterion
        self.shot = shot
        self.vgg = vgg
        self.ppm_scales = ppm_scales
        self.pyramid_bins = self.ppm_scales
        models.BatchNorm = BatchNorm

        # Backbone Related
        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)
        else:
            print('INFO: Using ResNet {}'.format(layers))
            if layers == 50:
                resnet = models.resnet50(pretrained=pretrained)
            elif layers == 101:
                resnet = models.resnet101(pretrained=pretrained)
            else:
                resnet = models.resnet152(pretrained=pretrained)
            # stage 0
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                        resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            # stage 1-4 from res-50
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4


        # feature dimension
        reduce_radio = 16
        if self.vgg:
            reduce_dim = 128
        else:
            reduce_dim = 256

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + 1, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        self.avgpool_mask = nn.AdaptiveAvgPool2d(1)

        self.MPR = MPR(reduce_dim, int(reduce_dim / 2)).cuda()
        self.MPE = MPE(reduce_dim, reduce_radio)
        self.fusion = nn.Sequential(
            nn.Conv2d(reduce_dim + 1, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )




    def forward(self, q, s, s_mask, q_mask):
        """
            :param q: query image [b, 3, 200, 200]
            :param s: support image [b, shot, 3, 200, 200]
            :param n: normal image [b, shot, 3, 200, 200]
            :param s_mask: support mask [b, shot, 200, 200]
            :param q_mask: query mask [b, 200, 200]
            :return: out if test else out,loss
        """

        h, w = q.size(2), q.size(3)
        query_feat_0,query_feat_1, query_feat_2, query_feat_3 = self.query_encoder(q)

        # n-shot
        foreground_feat_list_0, foreground_feat_list_1, foreground_feat_list_2, foreground_feat_list_3, mask_1 = \
            self.support_encoder(s, s_mask)
        supp_feat_0 = foreground_feat_list_0[0]
        if self.shot > 1:
            for i in range(1, len(foreground_feat_list_0)):
                supp_feat_0 += foreground_feat_list_0[i]
            supp_feat_0 /= len(foreground_feat_list_0)
        supp_feat_1 = foreground_feat_list_1[0]
        if self.shot > 1:
            for i in range(1, len(foreground_feat_list_1)):
                supp_feat_1 += foreground_feat_list_1[i]
            supp_feat_1 /= len(foreground_feat_list_1)
        supp_feat_2 = foreground_feat_list_2[0]
        if self.shot > 1:
            for i in range(1, len(foreground_feat_list_2)):
                supp_feat_2 += foreground_feat_list_2[i]
            supp_feat_2 /= len(foreground_feat_list_2)
        supp_feat_3 = foreground_feat_list_3[0]
        if self.shot > 1:
            for i in range(1, len(foreground_feat_list_3)):
                supp_feat_3 += foreground_feat_list_3[i]
            supp_feat_3 /= len(foreground_feat_list_3)

        # MPR
        p_main = self.MPR(supp_feat_1, query_feat_1)

        # MPE
        p_proj = self.avgpool_mask(supp_feat_1).expand(-1, -1, query_feat_1.size(2), query_feat_1.size(3))
        p_aux = self.CBAM(query_feat_1 * p_proj)

        # IFM
        s = F.cosine_similarity(query_feat_1, supp_feat_1).unsqueeze(1)
        query_feat = torch.cat([p_main, p_aux, s], 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat
        out = self.cls(query_feat)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(out, q_mask.long())
            return out.max(1)[1], main_loss
        else:
            return out

    def query_encoder(self, x):

        query_feat_0 = self.layer0(x)  # [128,50,50]
        query_feat_1 = self.layer1(query_feat_0)  # [256,25,25]
        query_feat_2 = self.layer2(query_feat_1)  # [512,13,13]
        query_feat_3 = self.layer3(query_feat_2)  # [1024,6,6]

        if self.vgg:
            query_feat_3 = F.interpolate(query_feat_3, size=(query_feat_2.size(2), query_feat_2.size(3)),
                                         mode='bilinear', align_corners=True)

        return query_feat_0,query_feat_1,query_feat_2,query_feat_3

    def support_encoder(self, s_x, s_y):
        mask_list = []  # [shot,b,1,200,200]
        foreground_feat_list_0 = []
        foreground_feat_list_1 = []
        foreground_feat_list_2 = []
        foreground_feat_list_3 = []

        # background_feat_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_list.append(mask)

            supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
            supp_feat_1 = self.layer1(supp_feat_0)
            supp_feat_2 = self.layer2(supp_feat_1)
            supp_feat_3 = self.layer3(supp_feat_2)
            mask_0 = F.interpolate(mask, size=(supp_feat_0.size(2), supp_feat_0.size(3)), mode='bilinear',
                                 align_corners=True)
            mask_1 = F.interpolate(mask, size=(supp_feat_1.size(2), supp_feat_1.size(3)), mode='bilinear',
                                 align_corners=True)
            mask_2 = F.interpolate(mask, size=(supp_feat_2.size(2), supp_feat_2.size(3)), mode='bilinear',
                                 align_corners=True)
            if self.vgg:
                supp_feat_3 = F.interpolate(supp_feat_3, size=(supp_feat_2.size(2), supp_feat_2.size(3)),
                                            mode='bilinear', align_corners=True)

            mask_3 = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                   align_corners=True)
            supp_feat_0 = supp_feat_0 * mask_0
            supp_feat_1 = supp_feat_1 * mask_1
            supp_feat_2 = supp_feat_2 * mask_2
            supp_feat_3 = supp_feat_3 * mask_3
            foreground_feat_list_0.append(supp_feat_0)
            foreground_feat_list_1.append(supp_feat_1)
            foreground_feat_list_2.append(supp_feat_2)
            foreground_feat_list_3.append(supp_feat_3)

        return foreground_feat_list_0,foreground_feat_list_1,foreground_feat_list_2,foreground_feat_list_3, mask_1
