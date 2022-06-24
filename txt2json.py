from io import BytesIO
import os
import json
from PIL import Image, ImageDraw, ImageFont
import base64
import cv2
import numpy as np


class XmlMaker:
    def __init__(self, txtpath):
        self.txtPath = txtpath

    def get_imgData(self, img_name):
        img_name = Image.fromarray(np.uint8(img_name))
        img_buffer = BytesIO()
        img_name.save(img_buffer, format='JPEG')
        byte_data = img_buffer.getvalue()
        base64_data = base64.b64encode(byte_data).decode()
        # with open(img_name, "rb")as f:
        #     base64_data = str(base64.b64encode(f.read()))
        # match_pattern = re.compile(r'b\'(.*)\'')
        # base64_data = match_pattern.match(base64_data).group(1)
        return base64_data

    def yolov5txt_to_json(self, save_path, W=800, H=2048, img_path=None):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # '建立json文件'
        data = {}
        data['version'] = "1.0.0"
        data['flags'] = {}
        data['shapes'] = []
        name = self.txtPath.strip().split('/')[-1]
        filename = name[:-4] + '.jpg'
        if save_path:
            try:
                # emg = Image.open(os.path.join(img_path, filename))
                emg = Image.fromarray(np.uint8(img_path))
            except FileNotFoundError as e:
                return
                # os.remove(img_path.replace('jpg', 'txt'))
            W, H = emg.size[0], emg.size[1]
        assert name[-4:] == '.txt', "only read .txt"
        data['imagePath'] = filename
        # data['imageData'] = "null"
        # data['imageData'] = self.get_imgData(os.path.join(img_path, filename))
        data['imageData'] = self.get_imgData(img_path)

        # 如果存在预测框文件
        if os.path.exists(self.txtPath):
            txtfile = open(self.txtPath, 'r')
            txtList = txtfile.readlines()
            for i in txtList:
                print(i)
                x1 = float(i.strip().split(" ")[1])
                y1 = float(i.strip().split(" ")[2])
                x2 = float(i.strip().split(" ")[3])
                y2 = float(i.strip().split(" ")[4])
                label = {}
                label['points'] = []
                label['group_id'] = None
                label['shape_type'] = "rectangle"
                label['flags'] = {}
                label['label'] = class_names[int(i.strip().split(" ")[0])]
                label['points'].append([float(x1 * W - x2 * W / 2), float(y1 * H - y2 * H / 2)])
                label['points'].append([float(x1 * W + x2 * W / 2), float(y1 * H + y2 * H / 2)])
                # up['points']['xmin'] = int(x1 * W - x2 * W / 2)
                # up['points']['ymin'] = int(y1 * H - y2 * H / 2)
                # up['points']['xmax'] = int(x1 * W + x2 * W / 2)
                # up['points']['ymax'] = int(y1 * H + y2 * H / 2)
                data['shapes'].append(label)
                data['imageHeight'] = H
                data['imageWidth'] = W
        else:
            data['imageHeight'] = H
            data['imageWidth'] = W

        # data['time_labeled'] = str(100)
        # data['labeled'] = 'true'
        # A3 = {}  # {"width":3400,"height":587,"depth":3}
        # A3['width'], A3['height'], A3['depth'] = W, H, str(3)
        # data['size'] = A3
        article = json.dumps(data, ensure_ascii=False)
        f = open(
            os.path.join(save_path, name[:-4] + '.json'), 'w')
        f.write(article)

    def yolov5json_to_txt(self, save_path, areas=False, select=False, img_path=None):
        WF = open(os.path.join(self.txtPath), encoding='UTF-8')
        ZTXT = json.load(WF)  # 加载json文件内容

        print(ZTXT)
        filename = ZTXT["path"]  # 获取json文件里面path所对应的东西
        filename_V5 = filename.strip().split('\\')  # 按\\将路径单独分开
        jpg = filename_V5[-1]  # 只取出照片的名字**.png
        emg = Image.open(os.path.join(img_path, jpg))
        W_1, H_1 = emg.size[0], emg.size[1]
        LABE = os.path.join(save_path, '{}'.format(jpg[:-4] + '.txt'))
        LABE_file = open(LABE, "w")
        objects = ZTXT['outputs']['object']  # 取出json文件里面outputs下object里面的东西

        print(objects)
        W = int(ZTXT['size']['width'])
        H = int(ZTXT['size']['height'])  # 读取图片的宽度与高度
        for i, object in enumerate(objects):  # enumerate既遍历索引又遍历元素

            if object['name'] not in class_names:
                print('未知类别--{}--'.format(object['name']))
                continue
            cla = class_names.index(str(object['name']))

            xmin = int(float(object['bndbox']['xmin']))
            ymin = int(float(object['bndbox']['ymin']))
            xmax = int(float(object['bndbox']['xmax']))
            ymax = int(float(object['bndbox']['ymax']))
            x_1 = (xmin + xmax) / float(2 * W)
            y_1 = (ymin + ymax) / float(2 * H)
            x_w = float(xmax - xmin)
            x_h = float(ymax - ymin)

            # '做统计用'
            area = x_w, x_h
            areas.append(area)  # 将area不动追加到sareas末尾

            LABE_file.write(
                "{} {} {} {} {}\n".format(cla, x_1, y_1, (xmax - xmin) / float(W), (ymax - ymin) / float(H)))


if __name__ == "__main__":

    class_names = ['polyp']
    # 待检测视频路径
    video_path = r'/data/anhui-ai/lhq/video/'
    # 检测生成的txt路径
    label_path = r'/data/anhui-ai/lhq/yolov5/runs/detect/exp/labels'
    # json保存路径
    save_path = r'/data/anhui-ai/lhq/yolov5/runs/detect/exp/labels'

    for video in os.listdir(video_path):
        print(video)
        cap = cv2.VideoCapture(os.path.join(video_path, video))
        isOpened = cap.isOpened
        print(isOpened)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(fps, width, height)

        i = 0
        while(isOpened):
            i = i+1
            (flag, frame) = cap.read()  #读取每一帧，flag表示是否读取成功，frame为图片内容。
            # 修改
            read = XmlMaker(os.path.join(label_path, video[:-4]+'_'+str(i)+'.txt'))
            read.yolov5txt_to_json(save_path, W=width, H=height, img_path=frame)                

        print("end!")

