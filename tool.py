# 一些工具函数定义
import os
import time
from pathlib import Path
import cv2
from PIL import Image
from matplotlib import pyplot as plt
# 视频帧转视频功能
def ptov(name):
    imgPath= "data/images/"
    images = os.listdir(imgPath)
    fps = 20  # 帧率
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    im = Image.open(imgPath + images[10])
    videoWriter = cv2.VideoWriter(name, fourcc, fps, im.size)
    for im_name in range(10, len(images)-10):
        frame = cv2.imread(imgPath + images[im_name])
        videoWriter.write(frame)
    videoWriter.release()
# 视频解码函数，用于从视频中抽取帧并保存为图像
def decode_video(video_path, save_dir, target_num=None, centercrop=False):
    '''
    video_path: 待解码的视频
    save_dir: 抽帧图片的保存文件夹
    target_num: 抽帧的数量, 为空则解码全部帧, 默认抽全部帧
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("can not open the video")
        exit(1)
    count = 0
    index = 0
    frames_num = video.get(7)
    # 如果target_num为空就全部抽帧,不为空就抽target_num帧
    if target_num is None:
        step = 1
        print('all frame num is {}, decode all'.format(int(frames_num)))
    else:
        step = int(frames_num / target_num)
        print('all frame num is {}, decode sample num is {}'.format(int(frames_num), int(target_num)))
    while True:
        _, frame = video.read()
        if frame is None:
            break
        if count % step == 0:
            save_path = "{}/{:>06d}.jpg".format(save_dir, index)
            if centercrop:
                shape = frame.shape
                if(shape[2] == 3):
                    if(shape[1]>shape[0]):
                        s = int((shape[1]-shape[0])/2)
                        frame = frame[:, s:-s, :]
                    elif (shape[1]<shape[0]):
                        s = int((shape[0] - shape[1]) / 2)
                        frame = frame[s:-s, :, :]

            cv2.imwrite(save_path, frame)
            index += 1
        count += 1
        if index == frames_num and target_num is None:
            # 如果全部抽，抽到所有帧的最后一帧就停止
            break
        elif index == target_num and target_num is not None:
            # 如果采样抽，抽到target_num就停止
            break
        else:
            pass
    video.release()
# 图片序列转视频
def image_to_video(sample_dir=None, video_name="1022chen"):
    # 删除视频的最后20帧
    ls = os.listdir(sample_dir)
    for i in ls[-20:]:
        c_path = os.path.join(sample_dir, i)
        os.remove(c_path)
    command = 'ffmpeg -framerate 20  -i ' + sample_dir + '/%06d.jpg -c:v libx264 -y -vf format=yuv420p ' + video_name
    # ffmpeg -framerate 25 -i real_%d.png -c:v libx264 -y -vf format=yuv420p real.mp4
    print(command)
    os.system(command)

if __name__=='__main__':
    # image_path = "data/images"
    # media_path = "data/videos/1022chen.mp4"
    # image_to_video(sample_dir=image_path, video_name=media_path)
    # decode_video(media_path, "data/imgs/", target_num=None)
    import uuid
    uuid_str = uuid.uuid4().hex
    tmp_file_name = 'tmpfile_%s.mp4' % uuid_str
    print(tmp_file_name)


