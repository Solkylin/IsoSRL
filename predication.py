#单个视频预测代码
import torch
from PIL import Image
from torchvision import transforms
from models import r2plus1d_18, r3d_18, ResCRNN
import os
import cv2
from tool import decode_video
# LSTM模型的参数配置
lstm_hidden_size = 512
lstm_num_layers = 1
attention = False
# 从文件夹读取图片并进行预处理
def read_images(folder_path, frames=16):
    assert len(os.listdir(folder_path)) >= frames, "Too few images in your data folder: " + str(
        folder_path)
    images = []
    start = 1
    step = int(len(os.listdir(folder_path)) / frames) # 计算步长以均匀采样

    for i in range(frames):
        index = "{:06d}.jpg".format(start + i * step)
        image = Image.open(os.path.join(folder_path, index))  # 打开图片
        image = transform(image) # 应用预处理
        images.append(image)

    images = torch.stack(images, dim=0) # 堆叠为tensor
    images = images.permute(1, 0, 2, 3) # 调整维度适应3D CNN
    return images
# 将标签转换为对应的单词
def label_to_word(label):
    labels = {}
    try:
        label_file = open(label_path, 'r', encoding='utf-8')
        for line in label_file.readlines():
            line = line.strip()
            line = line.split('\t')
            labels[line[0]] = line[1]
    except Exception as e:
        raise

    if isinstance(label, torch.Tensor):
        return labels['{:06d}'.format(label.item())]
    elif isinstance(label, int):
        return labels['{:06d}'.format(label)]
# 定义预测函数
def predict(file_name, selected_model, selected_weight, agree, video_style, centercrop):
    # 根据视频类型确定路径
    if video_style == "CSL测试集":
        video_path = f'data/ptov/{file_name}'
    else:
        video_path = f'data/videos/{file_name}'
    data_path = './video/images/{}'.format(file_name[:-4])
    decode_video(video_path, data_path, centercrop=centercrop)
    # 移除背景
    if agree:
        pass
        # removebg(data_path)
    # data_path = "./video/images/kt"
    num_classes = int(selected_model[-3:])

    if (selected_model=="r2+1d_100") | (selected_model=="r2+1d_500"):
        model_path  = "weight/"+str(selected_model)+"/" +str(selected_weight)
        model = r2plus1d_18(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()

    elif (selected_model=="r3d_100") | (selected_model=="r3d_500"):
        model_path = "weight/" + str(selected_model) + "/" +str(selected_weight)
        model = r3d_18(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()

    elif (selected_model=="LSTM_100") | (selected_model=="LSTM_500"):
        model_path = "weight/" + str(selected_model) + "/" +str(selected_weight)
        sample_size = 128
        sample_duration = 16  # 抽帧
        lstm_hidden_size = 512
        lstm_num_layers = 1
        attention = False
        model = ResCRNN(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes,
                        lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers, arch="resnet18",
                        attention=attention)
        model.load_state_dict(torch.load(model_path))
        model.eval()

    images = read_images(data_path) # 读取处理好的图像
    # shutil.rmtree(data_path)
    # print(images.shape)
    images = torch.reshape(images, (1, 3, 16, 128, 128)) # 调整图像尺寸
    with torch.no_grad():
        outputs = model(images)
        if isinstance(outputs, list):
             outputs = outputs[0]

        prob_tensor = torch.nn.Softmax(dim=1)(outputs) # 应用softmax
        top_k = torch.topk(prob_tensor, 5, dim=1) # 取最高的5个结果
        probabilites = top_k.values.detach().numpy().flatten()
        indices = top_k.indices.detach().numpy().flatten()
        formatted_predictions = []

        for pred_prob, pred_idx in zip(probabilites, indices):
            predicted_perc = pred_prob * 100
            formatted_predictions.append(
                (label_to_word(int(pred_idx)), f"{predicted_perc:.3f}%"))

        ida = indices[0]
        return ida, formatted_predictions
        # prediction = torch.max(outputs, 1)[1]
        # return prediction, label_to_word(prediction)

transform = transforms.Compose([transforms.Resize([128, 128]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

label_path = 'data/dictionary.txt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    import time
    start_time = time.time()
    video_path = 'video/test_064.mp4'
    save_dir = 'video/images/test_064'
    decode_video(video_path, save_dir)

    model_path = "weight/r2+1d_100/r2+1d18_epoch012.pth"
    sample_size = 128
    sample_duration = 16
    num_classes = 100

    model = r2plus1d_18(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()


    # removebg(save_dir)
    images = read_images(save_dir)

    print(images.shape)
    images = torch.reshape(images, (1, 3, 16, 128, 128))
    with torch.no_grad():
        outputs = model(images)

        if isinstance(outputs, list):
             outputs = outputs[0]
        prob_tensor = torch.nn.Softmax(dim=1)(outputs)
        top_k = torch.topk(prob_tensor, 5, dim=1)
        probabilites = top_k.values.detach().numpy().flatten()
        indices = top_k.indices.detach().numpy().flatten()
        formatted_predictions = []

        for pred_prob, pred_idx in zip(probabilites, indices):
            predicted_perc = pred_prob * 100
            formatted_predictions.append(
                (label_to_word(int(pred_idx)), f"{predicted_perc:.3f}%"))

        ida = indices[0]
        print(ida)
        print(formatted_predictions)
    end_time = time.time()  # 程序结束时间
    run_time = end_time - start_time  # 程序的运行时间，单位为秒
    print(run_time)
