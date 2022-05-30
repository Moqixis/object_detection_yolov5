# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
PyTorch utils
"""

import math
import os
import platform
import subprocess
import time
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from utils.general import LOGGER, file_update_date, git_describe

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', message='User provided device_type of \'cuda\', but CUDA is not available. Disabling')


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    # Decorator to make all processes in distributed training wait for each local_master to do something
    """用在train.py
    用于处理模型进行分布式训练时同步问题
    基于torch.distributed.barrier()函数的上下文管理器，为了完成数据的正常同步操作（yolov5中拥有大量的多线程并行操作）
    Decorator to make all processes in distributed training wait for each local_master to do something.
    :params local_rank: 代表当前进程号  0代表主进程  1、2、3代表子进程
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def device_count():
    # Returns number of CUDA devices available. Safe version of torch.cuda.device_count(). Only works on Linux.
    assert platform.system() == 'Linux', 'device_count() function only works on Linux'
    try:
        cmd = 'nvidia-smi -L | wc -l'
        return int(subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1])
    except Exception:
        return 0


def select_device(device='', batch_size=0, newline=True):
    """广泛用于train.py、val.py、detect.py等文件中
    用于选择模型训练的设备 并输出日志信息
    :params device: 输入的设备  device = 'cpu' or '0' or '0,1,2,3'
    :params batch_size: 一个批次的图片个数
    """
    # device = 'cpu' or '0' or '0,1,2,3'
    # git_describe(): 返回当前文件父文件的描述信息(yolov5)   date_modified(): 返回当前文件的修改日期
    # s: 之后要加入logger日志的显示信息
    s = f'YOLOv5 🚀 {git_describe() or file_update_date()} torch {torch.__version__} '  # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        # 如果cpu=True 就强制(force)使用cpu 令torch.cuda.is_available() = False
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        # 如果输入device不为空  device=GPU  直接设置 CUDA environment variable = device 加入CUDA可用设备
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        # 检查cuda的可用性 如果不可用则终止程序
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    # 输入device为空 自行根据计算机情况选择相应设备  先看GPU 没有就CPU
    # 如果cuda可用 且 输入device != cpu 则 cuda=True 反正cuda=False
    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        # devices: 如果cuda可用 返回所有可用的gpu设备 i.e. 0,1,6,7  如果不可用就返回 '0'
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        # n: 所有可用的gpu设备数量  device count
        n = len(devices)  # device count
        # 检查是否有gpu设备 且 batch_size是否可以能被显卡数目整除  check batch_size is divisible by device_count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            # 如果不能则关闭程序
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)  # 定义等长的空格
        # 满足所有条件 s加上所有显卡的信息
        for i, d in enumerate(devices):
            # p: 每个可用显卡的相关属性
            p = torch.cuda.get_device_properties(i)
            # 显示信息s加上每张显卡的属性信息
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
    else:
        # cuda不可用显示信息s就加上CPU
        s += 'CPU\n'

    if not newline:
        s = s.rstrip()
    # 将显示信息s加入logger日志文件中
    LOGGER.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    # 如果cuda可用就返回第一张显卡的的名称 如: GeForce RTX 2060 反之返回CPU对应的名称
    return torch.device('cuda:0' if cuda else 'cpu')


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(input, ops, n=10, device=None):
    """
    输出某个网络结构(操作ops)的一些信息: 总参数 浮点计算量 前向传播时间 反向传播时间 输入变量的shape 输出变量的shape
    :params x: 输入tensor x
    :params ops: 操作ops(某个网络结构)
    :params n: 执行多少轮ops
    :params device: 执行设备
    """
    # YOLOv5 speed/memory/FLOPs profiler
    #
    # Usage:
    #     input = torch.randn(16, 3, 640, 640)
    #     m1 = lambda x: x * torch.sigmoid(x)
    #     m2 = nn.SiLU()
    #     profile(input, [m1, m2], n=100)  # profile over 100 iterations

    results = []
    device = device or select_device()
    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s}")

    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, 'to') else m  # device
            m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # GFLOPs
            except Exception:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception:  # no backward method
                        # print(e)  # for debug
                        t[2] = float('nan')
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else 'list'
                s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else 'list'
                p = sum(list(x.numel() for x in m.parameters())) if isinstance(m, nn.Module) else 0  # parameters
                print(f'{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}')
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results


def is_parallel(model):
    """在ModelEMA类中调用
    用于判断模型是否支持并行  Returns True if model is of type DP or DDP
    """
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    """用在train.py中, 用于加载和保存模型(参数)
    判断单卡还是多卡(能否并行)  多卡返回model.module  单卡返回model
    """
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    # 如果model支持并行(多卡)就返回model.module  不支持并行就返回model
    # 用在tain中保存模型 因为多卡训练的时候直接用model.state_dict()进行保存的模型, 每个层参数的名称前面会加上module,
    # 这时候再用单卡(gpu) model_dict加载model.state_dict()参数时会出现名称不匹配的情况,
    # 因此多卡保存模型时注意使用model.module.state_dict() 即返回model.module  单卡返回model即可
    return model.module if is_parallel(model) else model


def initialize_weights(model):
    """在yolo.py的Model类中的init函数被调用
    用于初始化模型权重
    """
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    """
    用于找到模型model中类型是mclass的层结构的索引  Finds layer indices matching module class 'mclass'
    :params model: 模型
    :params mclass: 层结构类型 默认nn.Conv2d
    """
    # Finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    """在prune中调用
    用于求模型model的稀疏程度sparsity   Return global model sparsity
    """
    # Return global model sparsity
    # 初始化模型的总参数个数a(前向+反向)  模型参数中值为0的参数个数b
    a, b = 0, 0
    # model.parameters()返回模型model的参数 返回一个生成器 需要用for循环或者next()来获取参数
    # for循环取出每一层的前向传播和反向传播的参数
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    # b / a 即可以反应模型的稀疏程度
    return b / a


def prune(model, amount=0.3):
    """可以用于test.py和detect.py中进行模型剪枝
    对模型model进行剪枝操作 以增加模型的稀疏性  使用prune工具将参数稀疏化
    https://github.com/ultralytics/yolov5/issues/304
    :params model: 模型
    :params amount: 随机裁剪(总参数量 x amount)数量的参数
    """
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    # 模型的迭代器 返回的是所有模块的迭代器  同时产生模块的名称(name)以及模块本身(m)
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            # 对当前层结构m, 随机裁剪(总参数量 x amount)数量的权重(weight)参数
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            # 彻底移除被裁剪的的权重参数
            prune.remove(m, 'weight')  # make permanent
    # 输出模型的稀疏度 调用sparsity函数计算当前模型的稀疏度
    print(' %.3g global sparsity' % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    """在yolo.py中Model类的fuse函数中调用
    融合卷积层和BN层(测试推理使用)   Fuse convolution and batchnorm layers
    方法: 卷积层还是正常定义, 但是卷积层的参数w,b要改变   通过只改变卷积参数, 达到CONV+BN的效果
          w = w_bn * w_conv   b = w_bn * b_conv + b_bn   (可以证明)
    https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    https://github.com/ultralytics/yolov3/issues/807
    https://zhuanlan.zhihu.com/p/94138640
    :params conv: torch支持的卷积层
    :params bn: torch支持的bn层
    """
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    # w_conv: 卷积层的w参数 直接clone conv的weight即可
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    # w_bn: bn层的w参数(可以自己推到公式)  torch.diag: 返回一个以input为对角线元素的2D/1D 方阵/张量?
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    # w = w_bn * w_conv      torch.mm: 对两个矩阵相乘
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    # b_conv: 卷积层的b参数 如果不为None就直接读取conv.bias即可
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    # b_bn: bn层的b参数(可以自己推到公式)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    #  b = w_bn * b_conv + b_bn   (w_bn not forgot)
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, img_size=640):
    """用于yolo.py文件的Model类的info函数
    输出模型的所有信息 包括: 所有层数量, 模型总参数量, 需要求梯度的总参数量, img_size大小的model的浮点计算量GFLOPs
    :params model: 模型
    :params verbose: 是否输出每一层的参数parameters的相关信息
    :params img_size: int or list  i.e. img_size=640 or img_size=[640, 320]
    """
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPs
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPs
    except (ImportError, Exception):
        fs = ''

    name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(model, 'yaml_file') else 'Model'
    LOGGER.info(f"{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    """用于yolo.py文件中Model类的forward_augment函数中
    实现对图片的缩放操作
    :params img: 原图
    :params ratio: 缩放比例 默认=1.0 原图
    :params same_shape: 缩放之后尺寸是否是要求的大小(必须是gs=32的倍数)
    :params gs: 最大的下采样率 32 所以缩放后的图片的shape必须是gs=32的倍数
    """
    # Scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0: # 如果缩放比例ratio为1.0 直接返回原图
        return img
    else: # 如果缩放比例ratio不为1.0 则开始根据缩放比例ratio进行缩放
        # h, w: 原图的高和宽
        h, w = img.shape[2:]
        # s: 放缩后图片的新尺寸  new size
        s = (int(h * ratio), int(w * ratio))  # new size
        # 直接使用torch自带的F.interpolate(上采样下采样函数)插值函数进行resize
        # F.interpolate: 可以给定size或者scale_factor来进行上下采样
        #                mode='bilinear': 双线性插值  nearest:最近邻
        #                align_corner: 是否对齐 input 和 output 的角点像素(corner pixels)
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            # 缩放之后要是尺寸和要求的大小(必须是gs=32的倍数)不同 再对其不相交的部分进行pad
            # 而pad的值就是imagenet的mean
            # Math.ceil(): 向上取整  这里除以gs向上取整再乘以gs是为了保证h、w都是gs的倍数
            h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
        # pad img shape to gs的倍数 填充值为 imagenet mean
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    """在ModelEMA函数和yolo.py中Model类的autoshape函数中调用
    复制b的属性(这个属性必须在include中而不在exclude中)给a
    :params a: 对象a(待赋值)
    :params b: 对象b(赋值)
    :params include: 可以赋值的属性
    :params exclude: 不能赋值的属性
    """
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class EarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.')
        return stop


class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    用在train.py中的test.run（测试）阶段
    模型的指数加权平均方法(Model Exponential Moving Average)
    是一种给予近期数据更高权重的平均方法 利用滑动平均的参数来提高模型在测试数据上的健壮性/鲁棒性 一般用于测试集
    https://www.bilibili.com/video/BV1FT4y1E74V?p=63
    https://www.cnblogs.com/wuliytTaotao/p/9479958.html
    https://zhuanlan.zhihu.com/p/68748778
    https://zhuanlan.zhihu.com/p/32335746
    https://github.com/ultralytics/yolov5/issues/608
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model_ema.py
    
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """train.py
        model:
        decay: 衰减函数参数
               默认0.9999 考虑过去10000次的真实值
        updates: ema更新次数
        """
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates) # 随着更新次数 更新参数贝塔(d)

            # msd: 模型配置的字典 model state_dict  msd中的数据保持不变 用于训练
            msd = de_parallel(model).state_dict()  # model state_dict
            # 遍历模型配置字典 如: k=linear.bias  v=[0.32, 0.25]  ema中的数据发生改变 用于测试
            for k, v in self.ema.state_dict().items():
                # 这里得到的v: 预测值
                if v.dtype.is_floating_point:
                    v *= d  # 公式左边  decay * shadow_variable
                    # .detach() 使对应的Variables与网络隔开而不参与梯度更新
                    v += (1 - d) * msd[k].detach()  # 公式右边  (1−decay) * variable

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        # 调用上面的copy_attr函数 从model中复制相关属性值到self.ema中
        copy_attr(self.ema, model, include, exclude)
