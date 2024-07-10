# 模型定义
import numpy as np
import torch.hub
import torchvision.models.video
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

# Resnet+LSTM
class LSTMAttentionBlock(nn.Module):
    def __init__(self, hidden_size):
        super(LSTMAttentionBlock, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)

    def forward(self, hidden_states):

        score_first_part = self.fc1(hidden_states)

        h_t = hidden_states[:,-1,:]

        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(score, dim=1)

        context_vector = torch.bmm(hidden_states.permute(0,2,1), attention_weights.unsqueeze(2)).squeeze(2)

        pre_activation = torch.cat((context_vector, h_t), dim=1)
        attention_vector = self.fc2(pre_activation)
        attention_vector = torch.tanh(attention_vector)

        return attention_vector


class ResCRNN(nn.Module):
    def __init__(self, sample_size=256, sample_duration=16, num_classes=100,
                lstm_hidden_size=512, lstm_num_layers=1, arch="resnet18",
                attention=False):
        super(ResCRNN, self).__init__()
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes

        # network params
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.attention = attention

        # network architecture
        if arch == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif arch == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif arch == "resnet50":
            resnet = models.resnet50(pretrained=True)
        elif arch == "resnet101":
            resnet = models.resnet101(pretrained=True)
        elif arch == "resnet152":
            resnet = models.resnet152(pretrained=True)
        # delete the last fc layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.lstm = nn.LSTM(
            input_size=resnet.fc.in_features,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
        )
        if self.attention:
            self.attn_block = LSTMAttentionBlock(hidden_size=self.lstm_hidden_size)
        self.fc1 = nn.Linear(self.lstm_hidden_size, self.num_classes)

    def forward(self, x):
        # CNN
        cnn_embed_seq = []
        # x: (batch_size, channel, t, h, w)
        for t in range(x.size(2)):
            # with torch.no_grad():
            out = self.resnet(x[:, :, t, :, :])
            # print(out.shape)
            out = out.view(out.size(0), -1)
            cnn_embed_seq.append(out)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # print(cnn_embed_seq.shape)
        # batch first
        cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)

        # LSTM
        # use faster code paths
        self.lstm.flatten_parameters()
        out, (h_n, c_n) = self.lstm(cnn_embed_seq, None)
        # MLP
        if self.attention:
            out = self.fc1(self.attn_block(out))
        else:
            # out: (batch, seq, feature), choose the last time step
            out = self.fc1(out[:, -1, :])

        return out

class r3d_18(nn.Module):
    def __init__(self, pretrained=True, num_classes=500):
        super(r3d_18, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        model = torchvision.models.video.r3d_18(pretrained=self.pretrained)
        # delete the last fc layer
        modules = list(model.children())[:-1]
        # print(modules)
        self.r3d_18 = nn.Sequential(*modules)
        self.fc1 = nn.Linear(model.fc.in_features, self.num_classes)

    def forward(self, x):
        out = self.r3d_18(x)
        # print(out.shape)
        # Flatten the layer to fc
        out = out.flatten(1)
        out = self.fc1(out)

        return out


class r2plus1d_18(nn.Module):

    def __init__(self, pretrained=True, num_classes=500):
        super(r2plus1d_18, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        model = torchvision.models.video.r2plus1d_18(pretrained=pretrained)
        # delete the last fc layer
        modules = list(model.children())[:-1]
        self.r2plus1d_18 = nn.Sequential(*modules)
        self.fc1 = nn.Linear(model.fc.in_features, self.num_classes)
        # print(self.fc1)

    def forward(self, x):
        out = self.r2plus1d_18(x)
        # print(out.shape)
        # Flatten the layer to fc
        out = out.flatten(1)
        out = self.fc1(out)

        return out

class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)
        probs = self.softmax(logits)

        return probs


# class slowfast(nn.Module):
#     def __init__(self,  num_classes=100):
#         super().__init__()
#         self.num_classes = num_classes
#
#
#         model = create_slowfast(model_num_class=num_classes)
#         model_dict = model.state_dict()
#         pretrained_dict = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True).state_dict()
#         pretrained_dict = {
#                 k: v for k, v in pretrained_dict.items()
#                 if k in list(model_dict.keys()) and model_dict[k].shape == v.shape
#         }
#         model_dict.update(pretrained_dict)
#         model.load_state_dict(model_dict)
#         print('Loaded pretrained model.')
#         self.backbone = model
#
#     def forward(self, x):
#         x_fast = x  # shape [2, 3, 32, 224, 224]
#         x_slow = x_fast[:, :, ::4]  # shape [2, 3, 8, 224, 224]
#         # batch_size, n_frames, n_channels, height, width = x.size()  #   shape is [ batch, frames, 3, height, width ]
#         out = self.backbone([x_slow, x_fast])
#         return out
#
# def slowfast_r50(n_activations=2):
#     model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
#
#     # print(model.blocks[6])
#
#     # override last layer to fit the given prediction task
#     model.blocks[5].proj = nn.Linear(in_features=2048, out_features=n_activations, bias=True)
#     return model


# class mvit_v2_s(nn.Module):
#
#     def __init__(self, pretrained=True, num_classes=500):
#         super(mvit_v2_s, self).__init__()
#         self.pretrained = pretrained
#         self.num_classes = num_classes
#         model = torchvision.models.video.mvit_v2_s(pretrained=pretrained)
#         print(model)
#         # delete the last fc layer
#         modules = list(model.children())[:-1]
#         self.mvit_v2_s = nn.Sequential(*modules)
#
#         self.drop = nn.Dropout(p=0.5, inplace=True)
#         self.fc1 = nn.Linear(768, self.num_classes, bias=True)
#         #  print(self.fc1)
#
#     def forward(self, x):
#         out = self.mvit_v2_s(x)
#         # print(out.shape)
#         # Flatten the layer to fc
#         out = self.drop(out)
#         out = self.fc1(out)
#
#         return out

import torch
import torch.nn as nn
from transformers import ViTModel

class ViTBLSTM(nn.Module):
    def __init__(self, sample_size=256, sample_duration=16, num_classes=500,
                 lstm_hidden_size=512, lstm_num_layers=1,
                 attention=False, vit_name='vit-base-patch16-224'):
        super(ViTBLSTM, self).__init__()
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes

        # network params
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.attention = attention

        # Using a pre-trained Vision Transformer model
        self.vit = ViTModel.from_pretrained(vit_name)

        # Bidirectional LSTM
        self.blstm = nn.LSTM(
            input_size=self.vit.config.hidden_size,  # ViT hidden size
            hidden_size=self.lstm_hidden_size // 2,  # half size for each direction in BLSTM
            num_layers=self.lstm_num_layers,
            batch_first=True,
            bidirectional=True  # Enable bidirectional processing
        )
        if self.attention:
            self.attn_block = LSTMAttentionBlock(hidden_size=self.lstm_hidden_size)
        self.fc1 = nn.Linear(self.lstm_hidden_size, self.num_classes)  # lstm_hidden_size is the total size

    def forward(self, x):
        # ViT
        vit_embed_seq = []
        for t in range(x.size(2)):
            out = self.vit(x[:, :, t, :, :]).last_hidden_state
            out = out[:, 0, :]  # Use the [CLS] token
            vit_embed_seq.append(out)

        vit_embed_seq = torch.stack(vit_embed_seq, dim=0)
        vit_embed_seq = vit_embed_seq.transpose_(0, 1)

        # BLSTM
        self.blstm.flatten_parameters()
        out, (h_n, c_n) = self.blstm(vit_embed_seq, None)

        # MLP
        if self.attention:
            out = self.fc1(self.attn_block(out))
        else:
            out = self.fc1(out[:, -1, :])

        return out


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from dataset import CSL_Isolated

    ViTBLSTM = ViTBLSTM()
    print(ViTBLSTM)
    x = torch.randn(8, 3,  16, 224, 224)  # shape [2, 3, 32, 224, 224]
    logits = ViTBLSTM(x)  # shape [2, 1000]
    print(logits.shape)

    # x = torch.randn(16, 3, 32, 128, 128)  # shape [2, 3, 32, 224, 224]
    # model = slowfast()
    # print(model)r
    # logits = model(x)  # shape [2, 1000]
    # print(logits.shape)


    # sample_size = 128
    # sample_duration = 16
    # data_path = "../SLR_Dataset/CSL_Isolated/color_video_25000"
    # label_path = '../SLR_Dataset/CSL_Isolated/dictionary.txt'
    # num_classes = 100
    # transform = transforms.Compose([transforms.Resize([sample_size, sample_size]), transforms.ToTensor()])
    # dataset = CSL_Isolated(data_path, label_path, sample_duration, num_classes, transform=transform)
    #
    # cnn3d = r2plus1d_18(num_classes=num_classes)
    # # cnn3d = mvit_v2_s(num_classes=num_classes)
    #
    # print(cnn3d(dataset[0]['data']).shape)
