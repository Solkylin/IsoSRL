#模型测试代码
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from models import r3d_18, r2plus1d_18, ResCRNN
# 自定义数据集类，用于加载视频数据
class CSL_Isolated(Dataset):
    def __init__(self, data_path, label_path, frames=16, num_classes=500, train=True, transform=None):
        super(CSL_Isolated, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.train = train
        self.transform = transform
        self.frames = frames
        self.num_classes = num_classes
        self.signers = 50
        self.repetition = 5
        if self.train:
            self.videos_per_folder = int(0.8 * self.signers * self.repetition)
        else:
            self.videos_per_folder = int(0.2 * self.signers * self.repetition)
        self.data_folder = []
        try:
            obs_path = [os.path.join(self.data_path, item) for item in os.listdir(self.data_path)]
            self.data_folder = sorted([item for item in obs_path if os.path.isdir(item)])
        except Exception as e:
            print("Something wrong with your data path!!!")
            raise
        self.labels = {}
        try:
            label_file = open(self.label_path, 'r',encoding='utf-8')
            for line in label_file.readlines():
                line = line.strip()
                line = line.split('\t')
                self.labels[line[0]] = line[1]
        except Exception as e:
            raise
    # 从目录中读取图像，应用预处理，返回图像张量
    def read_images(self, folder_path):
        assert len(os.listdir(folder_path)) >= self.frames, "Too few images in your data folder: " + str(folder_path)
        images = []
        start = 1
        step = int(len(os.listdir(folder_path))/self.frames)

        for i in range(self.frames):
            index = "{:06d}.jpg".format(start + i * step)
            image = Image.open(os.path.join(folder_path, index))  #.convert('L')
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        # switch dimension for 3d cnn
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
        return images

    def __len__(self):
        return self.num_classes * self.videos_per_folder
    def __getitem__(self, idx):
        top_folder = self.data_folder[int(idx/self.videos_per_folder)]
        selected_folders = [os.path.join(top_folder, item) for item in os.listdir(top_folder)]
        selected_folders = sorted([item for item in selected_folders if os.path.isdir(item)])
        if self.train:
            selected_folder = selected_folders[idx%self.videos_per_folder]
        else:
            selected_folder = selected_folders[idx%self.videos_per_folder + int(0.8*self.signers*self.repetition)]
        images = self.read_images(selected_folder)
        # print(selected_folder, int(idx/self.videos_per_folder))
        # print(self.labels['{:06d}'.format(int(idx/self.videos_per_folder))])
        # label = self.labels['{:06d}'.format(int(idx/self.videos_per_folder))]
        label = torch.LongTensor([int(idx/self.videos_per_folder)])
        return {'data': images, 'label': label}
    def label_to_word(self, label):
        if isinstance(label, torch.Tensor):
            return self.labels['{:06d}'.format(label.item())]
        elif isinstance(label, int):
            return self.labels['{:06d}'.format(label)]
# 根据模型和数据加载器获取预测和真实标签，计算准确率
def get_label_and_pred(model, dataloader, device):
    all_label = []
    all_pred = []
    acc_label = []
    acc_pred = []
    accuracies = []
    with torch.no_grad():
        for batch_idx, data in enumerate(pre_loader):
            # get the inputs and labels
            inputs, labels = data['data'].to(device), data['label'].to(device)
            # forward
            outputs = model(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0]
            # collect labels & prediction
            prediction = torch.max(outputs, 1)[1]
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)
            acc_label.extend(labels.squeeze().cpu().numpy())
            acc_pred.extend(prediction.cpu().numpy())
            # Calculate accuracy for this batch
            correct = sum(1 for pred, label in zip(acc_pred, acc_label) if pred == label)
            total = len(acc_pred)
            accuracy = correct / total
            print(batch_idx, "批量准确率: ", accuracy)
            accuracies.append(accuracy)
            # Calculate average accuracy across all batches
        average_accuracy = sum(accuracies) / len(accuracies)
    # Compute accuracy
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    all_label = all_label.squeeze().cpu().data.squeeze().numpy()
    all_pred = all_pred.cpu().data.squeeze().numpy()
    return all_label, all_pred, average_accuracy
# 绘制混淆矩阵并保存结果
def plot_confusion_matrix(model, dataloader, device, save_path='confmat.png', normalize=True):
    # Get prediction
    all_label, all_pred, acc = get_label_and_pred(model, dataloader, device)
    print("测试集准确率：", acc)
    confmat = confusion_matrix(all_label, all_pred)

    # Normalize the matrix
    if normalize:
        confmat = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
    # Draw matrix
    plt.figure(figsize=(20,20))
    # confmat = np.random.rand(100,100)
    plt.imshow(confmat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    # Add ticks
    ticks = np.arange(100)
    plt.xticks(ticks, fontsize=8)
    plt.yticks(ticks, fontsize=8)
    plt.grid(True)
    # Add title & labels
    plt.title('Confusion matrix', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.ylabel('True label', fontsize=20)
    # Save figure
    plt.savefig(save_path)

    # Ranking
    sorted_index = np.diag(confmat).argsort()
    for i in range(100):
        # print(type(sorted_index[i]))
        print(pre_set.label_to_word(int(sorted_index[i])), confmat[sorted_index[i]][sorted_index[i]])
    # Save to csv
    np.savetxt('logs/confusion_matrix/'+model_name+'_matrix.csv', confmat, delimiter=',')
# 配置和初始化
batch_size = 16
sample_size = 128
sample_duration = 16
num_classes = 100
# 数据集的位置
data_path = r"C:\Sign\SLR_Dataset\CSL_Isolated\color_video_25000"
label_path = "../SLR_Dataset/CSL_Isolated/dictionary.txt"
# 权重的位置
model_path = "weight/r3d_100/r3d_epoch011.pth"
model_name = "R3D_100"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=='__main__':
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    pre_set = CSL_Isolated(data_path=data_path, label_path=label_path, frames=sample_duration,
                             num_classes=num_classes, train=False, transform=transform)
    pre_loader = DataLoader(pre_set, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)
    # 选择模型
    model = r3d_18(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    plot_confusion_matrix(model, pre_loader, device, save_path='logs/confusion_matrix/'+model_name+'_confmat.png', normalize=True)