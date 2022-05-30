# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python path/to/val.py --weights yolov5s.pt                 # PyTorch
                                      yolov5s.torchscript        # TorchScript
                                      yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s.xml                # OpenVINO
                                      yolov5s.engine             # TensorRT
                                      yolov5s.mlmodel            # CoreML (macOS-only)
                                      yolov5s_saved_model        # TensorFlow SavedModel
                                      yolov5s.pb                 # TensorFlow GraphDef
                                      yolov5s.tflite             # TensorFlow Lite
                                      yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    # gn = [w, h, w, h] 对应图片的宽高  用于后面归一化
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        # xyxy -> xywh 并作归一化处理
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        # 保存预测类别和坐标值到对应图片id.txt文件中
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

# 将预测信息保存到coco格式的json字典(后面存入json文件) Append to pycocotools JSON dictionary
def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    # 获取预测框 并将xyxy转为xywh格式
    box = xyxy2xywh(predn[:, :4])  # xywh
    # 获取预测框 并将xyxy转为xywh格式
    # 之前的的xyxy格式是左上角右下角坐标  xywh是中心的坐标和宽高
    # 而coco的json格式的框坐标是xywh(左上角坐标 + 宽高)
    # 所以这行代码是将中心点坐标 -> 左上角坐标  xy center to top-left corner
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    # image_id: 图片id 即属于哪张图片
    # category_id: 类别 coco91class()从索引0~79映射到索引0~90
    # bbox: 预测框坐标
    # score: 预测得分
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})

# ?不是很懂
def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    # niou为iou阈值的个数 iouv.shape[0]
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


@torch.no_grad() # 不参与反向传播
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
):
    """
    :params data: 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息 train.py时传入data_dict
    :params weights: 模型的权重文件地址 运行train.py=None 运行test.py=默认weights/yolov5s.pt
    :params batch_size: 前向传播的批次大小 运行test.py传入默认32 运行train.py则传入batch_size // WORLD_SIZE * 2
    :params imgsz: 输入网络的图片分辨率 运行test.py传入默认640 运行train.py则传入imgsz_test
    :params conf_thres: object置信度阈值 默认0.25
    :params iou_thres: 进行NMS时IOU的阈值 默认0.6
    :params task: 设置测试的类型 有train, val, test, speed or study几种 默认val
    :params device: 测试的设备
    :params single_cls: 数据集是否只用一个类别 运行test.py传入默认False 运行train.py则传入single_cls
    :params augment: 测试是否使用TTA Test Time Augment 默认False
    :params verbose: 是否打印出每个类别的mAP 运行test.py传入默认Fasle 运行train.py则传入nc < 50 and final_epoch
    :params save_txt: 是否以txt文件的形式保存模型预测框的坐标 默认False
    :params save_hybrid: 是否save label+prediction hybrid results to *.txt  默认False
                         是否将gt_label+pre_label一起输入nms
    :params save_conf: 是否保存预测每个目标的置信度到预测tx文件中 默认True
    :params save_json: 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签）
                       运行test.py传入默认Fasle 运行train.py则传入is_coco and final_epoch(一般也是False)
    :params project: 测试保存的源文件 默认runs/test
    :params name: 测试保存的文件地址 默认exp  保存在runs/test/exp下
    :params exist_ok: 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
    :params half: 是否使用半精度推理 FP16 half-precision inference 默认False
    :params model: 模型 如果执行test.py就为None 如果执行train.py就会传入ema.ema(ema模型)
    :params dataloader: 数据加载器 如果执行test.py就为None 如果执行train.py就会传入testloader
    :params save_dir: 文件保存路径 如果执行test.py就为‘’ 如果执行train.py就会传入save_dir(runs/train/expn)
    :params plots: 是否可视化 运行test.py传入默认True 运行train.py则传入plots and final_epoch
    :params wandb_logger: 网页可视化 类似于tensorboard 运行test.py传入默认None 运行train.py则传入wandb_logger(train)
    :params compute_loss: 损失函数 运行test.py传入默认None 运行train.py则传入compute_loss(train)
    :return (Precision, Recall, map@0.5, map@0.5:0.95, box_loss, obj_loss, cls_loss)
    """
    # ============================================== 1、初始化配置1 ==================================================
    # 初始化模型并选择相应的计算设备
    # 判断是否是训练时调用run函数(执行train.py脚本), 如果是就使用训练时的设备 一般都是train
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # 生成save_dir文件路径  
        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        # 检测输入图片的分辨率imgsz是否能被gs整除 只在运行test.py才需要自己生成check imgsz
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    # 计算mAP相关参数
    # 设置iou阈值 从0.5-0.95取10个(0.05间隔)   iou vector for mAP@0.5:0.95
    # iouv: [0.50000, 0.55000, 0.60000, 0.65000, 0.70000, 0.75000, 0.80000, 0.85000, 0.90000, 0.95000]
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    # mAP@0.5:0.95 iou个数=10个
    niou = iouv.numel()

    # 如果不是训练(执行val.py脚本调用run函数)就调用create_dataloader生成dataloader
    # 如果是训练(执行train.py调用run函数)就不需要生成dataloader 可以直接从参数中传过来testloader
    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights[0]} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad = 0.0 if task in ('speed', 'benchmark') else 0.5
        rect = False if task == 'benchmark' else pt  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0 # 初始化测试的图片的数量
    # 初始化混淆矩阵
    confusion_matrix = ConfusionMatrix(nc=nc)
    # dict{key(class_index):value(class_name)} 获取数据集所有类别的index和对应类名
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    # 设置tqdm进度条的显示信息
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    # 初始化p, r, f1, mp, mr, map50, map指标和时间dt t0, t1, t2
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # 初始化测试集的损失
    loss = torch.zeros(3, device=device)
    # 初始化json文件中的字典 统计信息 ap等
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    # 开始验证
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        t1 = time_sync()
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        # 如果half为True 就把图片变为half精度  uint8 to fp16/32
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1  # t0: 累计处理数据时间

        # out:       推理结果 1个 [bs, anchor_num*grid_w*grid_h, xywh+c+20classes] = [1, 19200+4800+1200, 25]
        # train_out: 训练结果 3个 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
        #                    如: [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
        # Inference
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        dt[1] += time_sync() - t2 # 累计前向推理时间 t1

        # compute_loss不为空 说明正在执行train.py  根据传入的compute_loss计算损失值
        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # 将真实框target的xywh(因为target是在labelimg中做了归一化的)映射到img(test)尺寸
        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        # save_hybrid: adding the dataset labels to the model predictions before NMS
        #              是在NMS之前将数据集标签targets添加到模型预测中
        # 这允许在数据集中自动标记(for autolabelling)其他对象(在pred中混入gt) 并且mAP反映了新的混合标签
        # targets: [num_target, img_index+class_index+xywh] = [31, 6]
        # lb: {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3  # 累计NMS时间

        # 统计每张图片的真实框、预测框信息  Statistics per image
        # 为每张图片做统计，写入预测信息到txt文件，生成json文件字典，统计tp等
        # out: list{bs}  [300, 6] [42, 6] [300, 6] [300, 6]  [pred_obj_num, x1y1x2y2+object_conf+cls]
        # Metrics
        for si, pred in enumerate(out):
            # 获取第si张图片的gt标签信息 包括class, x, y, w, h    target[:, 0]为标签属于哪张图片的编号
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            # 统计测试图片数量 +1
            seen += 1

            # 如果预测为空，则添加空的信息到stats里
            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((3, 0), device=device)))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            # 将预测坐标映射到原图img中
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots: # 混淆矩阵
                    confusion_matrix.process_batch(predn, labelsn)
            # 每张图片的结果统计到stats里
            # stats: correct, conf, pcls, tcls   bs个 correct, conf, pcls, tcls
            # correct: [pred_num, 10] bool 当前图片每一个预测框在每一个iou条件下是否是TP
            # pred[:, 4]: [pred_num, 1] 当前图片每一个预测框的conf
            # pred[:, 5]: [pred_num, 1] 当前图片每一个预测框的类别
            # tcls: [gt_num, 1] 当前图片所有gt框的class
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # 画出第1个batch的图片的ground truth和预测框并保存
        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            # Thread  表示在单独的控制线程中运行的活动 创建一个单线程(子线程)来执行函数 由这个子进程全权负责这个函数
            # target: 执行的函数  args: 传入的函数参数  daemon: 当主线程结束后, 由他创建的子线程Thread也已经自动结束了
            # .start(): 启动线程  当thread一启动的时候, 就会运行我们自己定义的这个函数plot_images
            # 如果在plot_images里面打开断点调试, 可以发现子线程暂停, 但是主线程还是在正常的训练(还是正常的跑)
            Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()
            # predictions 传入plot_images函数之前需要改变pred的格式  target则不需要改
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names), daemon=True).start()

        callbacks.run('on_val_batch_end')

    # 计算mAP
    # 统计stats中所有图片的统计结果 将stats列表的信息拼接到一起
    # stats(concat后): list{4} correct, conf, pcls, tcls  统计出的整个数据集的GT
    # correct [img_sum, 10] 整个数据集所有图片中所有预测框在每一个iou条件下是否是TP  [1905, 10]
    # conf [img_sum] 整个数据集所有图片中所有预测框的conf  [1905]
    # pcls [img_sum] 整个数据集所有图片中所有预测框的类别   [1905]
    # tcls [gt_sum] 整个数据集所有图片所有gt框的class     [929]
    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        # 精准度TP/TP+FP，召回率TP/P，map，f1分数，类别
        # 根据上面的统计预测结果计算p, r, ap, f1, ap_class（ap_per_class函数是计算每个类的mAP等指标的）等指标
        # p: [nc] 最大平均f1时每个类别的precision
        # r: [nc] 最大平均f1时每个类别的recall
        # ap: [71, 10] 数据集每个类别在10个iou阈值下的mAP
        # f1 [nc] 最大平均f1时每个类别的f1
        # ap_class: [nc] 返回数据集中所有的类别index  
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        # ap50: [nc] 所有类别的mAP@0.5   ap: [nc] 所有类别的mAP@0.5:0.95
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        # mp: [1] 所有类别的平均precision(最大f1时)
        # mr: [1] 所有类别的平均recall(最大f1时)
        # map50: [1] 所有类别的平均mAP@0.5
        # map: [1] 所有类别的平均mAP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        # nt: [nc] 统计出整个数据集的gt框中数据集各个类别的个数
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results  数据集图片数量 + 数据集gt框的数量 + 所有类别的平均precision +
    #                所有类别的平均recall + 所有类别的平均mAP@0.5 + 所有类别的平均mAP@0.5:0.95
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    # 细节展示每个类别的各个指标  类别 + 数据集图片数量 + 这个类别的gt框数量 + 这个类别的precision +
    #                        这个类别的recall + 这个类别的mAP@0.5 + 这个类别的mAP@0.5:0.95
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds t: {tuple: 3} 打印前向传播耗费的总时间、nms耗费总时间、总时间
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # 画出混淆矩阵并存入wandb_logger中
    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # 采用之前保存的json文件格式预测结果 通过cocoapi评估各个指标
    # 需要注意的是 测试集的标签也要转为coco的json格式
    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        # 获取预测框的json文件路径并打开
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # 获取并初始化测试集标签的json文件
            anno = COCO(anno_json)  # init annotations api
            # 初始化预测框的文件
            pred = anno.loadRes(pred_json)  # init predictions api
            # 创建评估器
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            # 评估
            eval.evaluate()
            eval.accumulate()
            # 展示结果
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    # (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()): {tuple:7}
    #      0: mp [1] 所有类别的平均precision(最大f1时)
    #      1: mr [1] 所有类别的平均recall(最大f1时)
    #      2: map50 [1] 所有类别的平均mAP@0.5
    #      3: map [1] 所有类别的平均mAP@0.5:0.95
    #      4: val_box_loss [1] 验证集回归损失
    #      5: val_obj_loss [1] 验证集置信度损失
    #      6: val_cls_loss [1] 验证集分类损失
    # maps: [80] 所有类别的mAP@0.5:0.95
    # t: {tuple: 3} 0: 打印前向传播耗费的总时间   1: nms耗费总时间   2: 总时间
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    """
    opt参数详解
    data: 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息
    weights: 模型的权重文件地址 weights/yolov5s.pt
    batch_size: 前向传播的批次大小 默认32
    imgsz: 输入网络的图片分辨率 默认640
    conf-thres: object置信度阈值 默认0.25
    iou-thres: 进行NMS时IOU的阈值 默认0.6
    task: 设置测试的类型 有train, val, test, speed or study几种 默认val
    device: 测试的设备
    single-cls: 数据集是否只用一个类别 默认False
    augment: 测试是否使用TTA Test Time Augment 默认False
    verbose: 是否打印出每个类别的mAP 默认False
    save-txt: 是否以txt文件的形式保存模型预测框的坐标 默认True
    save-hybrid: 是否save label+prediction hybrid results to *.txt  默认False 是否将gt_label+pre_label一起输入nms
    save-conf: 是否保存预测每个目标的置信度到预测tx文件中 默认True
    save-json: 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签） 默认False
    project: 测试保存的源文件 默认runs/test
    name: 测试保存的文件地址 默认exp  保存在runs/test/exp下
    exist-ok: 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
    half: 是否使用半精度推理 默认False
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/VOC.yaml', help='dataset.yaml path') # 修改
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp/weights/best.pt', help='model.pt path(s)') # 修改
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    # parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--task', default='test', help='train, val, test, speed or study') # 修改成test了
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()  # 解析上述参数
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    # 检测requirements文件中需要的包是否安装好了
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    # 如果task in ['train', 'val', 'test']就正常测试 训练集/验证集/测试集
    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        # 如果opt.task == 'speed' 就测试yolov5系列和yolov3-spp各个模型的速度评估
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        # 如果opt.task = ['study']就评估yolov5系列和yolov3-spp各个模型在各个尺度下的指标并可视化
        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            # 可视化各个指标
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
