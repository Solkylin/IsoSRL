#模型训练代码
import logging
import os
import torch
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import CSL_Isolated
from models import r3d_18, r2plus1d_18, ResCRNN
# 训练一个epoch的函数
def train_epoch(model, criterion, optimizer, dataloader, device, epoch, logger, log_interval, writer):
    model.train() # 设置模型为训练模式
    losses = []
    all_label = []
    all_pred = []

    print("train in device:" + str(device))

    for batch_idx, data in enumerate(dataloader):
        inputs, labels = data['data'].to(device), data['label'].to(device)
        optimizer.zero_grad() # 梯度清零
        # forward
        outputs = model(inputs)
        if isinstance(outputs, list):
            outputs = outputs[0]
        # compute loss
        loss = criterion(outputs, labels.squeeze())
        losses.append(loss.item())
        # compute accuracy
        prediction = torch.max(outputs, 1)[1]
        all_label.extend(labels.squeeze())
        all_pred.extend(prediction)
        score = accuracy_score(labels.squeeze().cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())
        # backward and optimize
        loss.backward()
        optimizer.step()
        # 每隔log_interval打印一次日志
        if (batch_idx + 1) % log_interval == 0:
            logger.info("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}%".format(epoch + 1, batch_idx + 1,
                                                                                           loss.item(), score * 100))
    # 计算平均损失和准确率
    training_loss = sum(losses) / len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    training_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(),
                                  all_pred.cpu().data.squeeze().numpy())
    # TensorBoard日志记录
    writer.add_scalars('Loss', {'train': training_loss}, epoch + 1)
    writer.add_scalars('Accuracy', {'train': training_acc}, epoch + 1)
    logger.info(
        "Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch + 1, training_loss, training_acc * 100))
# 验证一个epoch的函数
def val_epoch(model, criterion, dataloader, device, epoch, logger, writer):
    model.eval() # 设置模型为评估模式
    losses = []
    all_label = []
    all_pred = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # 获得inputs and labels
            inputs, labels = data['data'].to(device), data['label'].to(device)
            # forward
            outputs = model(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0]
            # compute the loss
            loss = criterion(outputs, labels.squeeze())
            losses.append(loss.item())
            # collect labels & prediction
            prediction = torch.max(outputs, 1)[1]
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)
    # 计算平均损失和准确率
    validation_loss = sum(losses) / len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    validation_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(),
                                    all_pred.cpu().data.squeeze().numpy())
    # TensorBoard日志记录
    writer.add_scalars('Loss', {'validation': validation_loss}, epoch + 1)
    writer.add_scalars('Accuracy', {'validation': validation_acc}, epoch + 1)
    logger.info("Average Validation Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch + 1, validation_loss,
                                                                                    validation_acc * 100))
# path
data_path = "../SLR_Dataset/CSL_Isolated/color_video_25000"
label_path = '../SLR_Dataset/CSL_Isolated/dictionary.txt'
model_path = "weight/r3d_100"
sum_path = "logs/r3d_100"
log_path = "logs/log/r3d.log"
# Log to file & tensorboard writer
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger('SLR')
# logger.info('Logging to file...')
writer = SummaryWriter(sum_path)
# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 超参数设置
num_classes = 100
epochs = 36
batch_size = 8
learning_rate = 1e-5
log_interval = 20  # 间隔log一次
sample_size = 128
sample_duration = 16  # 抽帧
lstm_hidden_size = 512
lstm_num_layers = 1
attention = True

if __name__ == '__main__':
    # load data
    transform = transforms.Compose([
        transforms.Resize([sample_size, sample_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_set = CSL_Isolated(data_path=data_path, label_path=label_path, frames=sample_duration,
                             num_classes=num_classes, train=True, transform=transform)
    val_set = CSL_Isolated(data_path=data_path, label_path=label_path, frames=sample_duration,
                           num_classes=num_classes, train=False, transform=transform)
    logger.info("Dataset samples: {}".format(len(train_set) + len(val_set)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=False)
    # model = CNN3D(sample_size=sample_size, sample_duration=sample_duration, drop_p=0,
    #             hidden1=512, hidden2=256, num_classes=num_classes).to(device)
    # model = ResCRNN(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes,
    #                 lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers, arch="resnet50",
    #                 attention=attention).to(device)
    model = r2plus1d_18(pretrained=True, num_classes=num_classes).to(device)
    # model = r3d_18(pretrained=True, num_classes=num_classes).to(device)
    # 训练中止后，继续train
    # model = r3d_18(num_classes=num_classes)
    # model.load_state_dict(torch.load("weight/r3d_100/r3d_epoch005.pth"))
    # model.to(device)
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # create loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = optim.SGD(model.parameters(), lr= 1e-3, momentum=0.9, weight_decay= 1e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 10, gamma=0.1)
    # train and val
    logger.info("Training Started".center(60, '#'))
    for epoch in range(epochs):
        train_epoch(model, criterion, optimizer, train_loader, device, epoch, logger, log_interval, writer)
        val_epoch(model, criterion, val_loader, device, epoch, logger, writer)
        torch.save(model.state_dict(), os.path.join(model_path, "test_epoch{:03d}.pth".format(epoch + 1)))
        logger.info("Epoch {} Model Saved".format(epoch + 1).center(60, '#'))
    logger.info("Training Finished".center(60, '#'))
