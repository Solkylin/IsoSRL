# streamlit run app.py
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
from predication import predict
import os
from real import play_webcam
from tool import image_to_video
# 设置页面配置
st.set_page_config(
    page_title="【润语无声】孤立词手语识别系统",
    page_icon="https://s21.ax1x.com/2024/04/05/pFb5BqS.png",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://baidu.com/',
        'Report a bug': "https://baidu.com/",
        'About': "# SLR by lcc"
    }
)
# 设置背景图片的CSS
def set_bg_img(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({url});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

bg_url = 'https://s21.ax1x.com/2024/04/05/pFboOHO.jpg'
set_bg_img(bg_url)


# 初始化或更新session_state
if 'show_model_info' not in st.session_state:
    st.session_state.show_model_info = False

# 初始化引导内容
if 'show_guide' not in st.session_state:
    st.session_state.show_guide = True  # 默认显示引导内容
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("手语识别系统")

# 添加新手引导
# 使用回调函数来改变状态
def toggle_guide():
    st.session_state.show_guide = not st.session_state.show_guide

# 判断当前状态，并显示相应的按钮
if st.session_state.show_guide:
    st.button('隐藏引导内容', on_click=toggle_guide)
else:
    st.button('显示引导内容', on_click=toggle_guide)

# 根据当前的状态决定是否显示引导内容
if st.session_state.show_guide:
    st.markdown("""
        ## 欢迎使用【润语无声】孤立词手语识别系统
        **【润语无声】孤立词手语识别系统** 旨在帮助用户通过视频识别手语动作，使得听障人士与他人的沟通更加便捷。此系统使用先进的深度学习技术，能够准确快速地识别各种手语动作，并提供相应的文字解释。
        
        ### 功能特点：
        - **上传视频识别**：用户可以上传手语视频，系统将自动识别视频中的手语动作，并展示识别结果。
        - **录制视频识别**：用户可以通过摄像头或者对屏幕进行录屏快速得到手语视频，系统将自动识别视频中的手语动作，并展示识别结果。        
        - **实时视频识别**：通过摄像头捕捉实时手语动作，系统实时给出识别反馈。
        - **测试集选择识别**：系统提供了内置的手语视频库供用户选择，便于快速了解常见手语动作。
        - **模型性能展示**：提供模型的准确率、损失等性能指标的详细信息，适合对深度学习感兴趣的用户。
        - **连续手语识别**：用户可以通过此按钮跳转至我们的测试版连续手语识别功能进行体验。

        ### 如何开始：
        请在侧边栏选择您希望进行的操作。如果您是首次使用，可以尝试选择内置的测试集视频库进行识别体验。
        
        ### 使用场景示例：
        本系统适用于学习手语的个人、希望与听障人士沟通的公众人员、以及手语研究教育工作者等。
                
        ### 使用教程：
        下附本系统的简易操作教程视频，跟随开发者的指引迅速上手本系统！
    """)
    # 插入图片
    # st.image("图片URL", caption="系统界面展示", use_column_width=True)

    # 插入视频（如果有的话）
    st.video("static/手语系统讲解.mp4")



with st.sidebar:
    st.header("配置")
    #model_options = ("r2+1d_100", "r3d_100", "LSTM_100", "r2+1d_500", "r3d_500", "LSTM_500")
    model_options = ("r2+1d_100", "r3d_100", "LSTM_100", "r2+1d_500", "r3d_500")
    selected_model = st.selectbox(
        label="选择使用的模型:",
        options=model_options,
    )
    page_weight = (
        os.listdir("weight/"+str(selected_model))
    )
    page_weight.sort(reverse=True)
    selected_weight = st.selectbox(
        label="选择训练的权重",
        options=page_weight,
    )
    video_style = st.selectbox(
        label="选择视频",
        options=("上传视频", "录制视频", "CSL测试集", "实时识别")
    )

if video_style == "上传视频":
    uploaded_file = st.sidebar.file_uploader("选择一个文件", type=['mp4', 'mp4v'])
    flag = False
elif video_style == "录制视频":
    flag = True
    uploaded_file = None
elif video_style == "实时识别":
    play_webcam(selected_model, selected_weight)
    flag = False
    uploaded_file = None
else:
    flag = False
    video_options=[f"{i:03}.mp4" for i in range(500)]
    selected_video = st.sidebar.selectbox(
        label="Choose a file",
        options=video_options
    )
    uploaded_file = "data/ptov/" + selected_video

with st.sidebar:

    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    # flag = st.checkbox('录制视频')

    # def recorder_factory():
    #     return MediaRecorder("data/videos/1022chen.mp4")

    # if flag:
    #     webrtc_streamer(
    #         key="demo",
    #         # media_stream_constraints={"video": True, "audio": True},
    #         in_recorder_factory=recorder_factory,
    #     )

    class VideoTransformer(VideoTransformerBase):

        def __init__(self):
            self.cnt = 0
        def transform(self, frame):

            img = frame.to_ndarray(format="bgr24")
            frame.to_image().save('data/images/%06d.jpg'%self.cnt)
            self.cnt += 1

            return img
    if flag:
        webrtc_streamer(key="example", video_processor_factory=VideoTransformer)


if (uploaded_file is not None) | flag:
    if (flag == False) & (video_style == "上传视频"):
        is_valid = True
        with st.spinner(text='资源加载中...'):
            st.sidebar.video(uploaded_file)
            with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
    else:
        is_valid = True
        if(video_style=="CSL测试集"):
            st.sidebar.video(uploaded_file)
else:
    is_valid = False
    # 显示/隐藏模型信息的按钮及其控制逻辑
    if st.sidebar.button('显示/隐藏模型信息'):
        st.session_state.show_model_info = not st.session_state.show_model_info

    if st.session_state.show_model_info:
        st.write(selected_model + "网络的展示    ---【润语无声】")
        tab1, tab2, tab3, tab4 = st.tabs(["acc曲线图", "loss曲线图", "网络可视化", "Confusion Matrix"])

        with tab1:
            df = pd.read_csv('logs/csv/{}_acc.csv'.format(selected_model))
            df.index = df.index + 1
            st.line_chart(df)

        with tab2:
            df = pd.read_csv('logs/csv/{}_loss.csv'.format(selected_model))
            df.index = df.index + 1
            st.line_chart(df)

        with tab3:
            display = open('logs/net/{}.onnx.svg'.format(selected_model), 'r', encoding='utf-8')
            source_code = display.read()
            components.html(source_code, height=1000, scrolling=True)

        with tab4:
            image = Image.open('logs/confusion_matrix/{}_confmat.png'.format(selected_model))
            st.image(image, caption='混淆矩阵')


xianshi = False

if is_valid:
    print('valid...')


    st1, st2 = st.columns(2)
    with st1:
        agree = st.checkbox('移除视频背景')
        if agree:
    
            st.write("移除背景模型运行在CPU上，可能需要几分钟！")
    with st2:
        centercrop = st.checkbox('裁剪居中')


    agree = False
    centercrop = False

    if st.button(':cake:开始识别'):


        # labels = predict(video_path)
        # st.write("Prediction index", labels[0], ", Prediction word: ", labels[1])

        # video_file = open(video_path, 'rb')
        # video_bytes = video_file.read()
        # st.video(video_bytes)

        if flag:    # 录制视频
            image_path = "data/images"

            import uuid
            uuid_str = uuid.uuid4().hex
            tmp_file_name = 'tmpfile_%s.mp4' % uuid_str

            media_path = "data/videos/"+tmp_file_name
            image_to_video(sample_dir=image_path, video_name=media_path)
            ida, prediction = predict(tmp_file_name, selected_model, selected_weight, agree, video_style, centercrop)
            xianshi = True
        else:
            if video_style == "上传视频":
                ida, prediction = predict(uploaded_file.name, selected_model, selected_weight, agree, video_style, centercrop)
            else:
                ida, prediction = predict(selected_video, selected_model, selected_weight, agree, video_style, centercrop)
        st.write("预测指标", ida)
        st.write("预测词: ", prediction[0][0])
        st.title("TOP-5 最可能的5个词语 ")
        df = pd.DataFrame(data=np.zeros((5, 2)),
                          columns=['词语', '可能性'],
                          index=np.linspace(1, 5, 5, dtype=int))

        # 跳转到手语词典搜索
        for idx, p in enumerate(prediction):
            ans = str(p[0])
            word = ans.split("（")
            link = 'https://www.spreadthesign.com/zh.hans.cn/search/?q=' + str(word[0])
            df.iloc[idx, 0] = f'<a href="{link}" target="_blank">{ans}</a>'
            df.iloc[idx, 1] = p[1]
        st.write(df.to_html(escape=False), unsafe_allow_html=True)

        st.balloons()


    else:
        st.write(f':point_up: Click :point_up:')

with st.sidebar:

    if flag & xianshi:
        st.video("data/videos/"+tmp_file_name)
        imgPath = "data/images/"
        ls = os.listdir(imgPath)
        for i in ls:
            c_path = os.path.join(imgPath, i)
            os.remove(c_path)

    # 使用HTML创建一个按钮，并设置onclick事件来实现页面跳转
    button_html = f"<html><body><a href='http://127.0.0.1:5001' target='_blank'><button>连续手语识别</button></a></body></html>"
    st.markdown(button_html, unsafe_allow_html=True)
    st.markdown(" ") # 换行符作用
    st.markdown("by 【润语无声】")
    