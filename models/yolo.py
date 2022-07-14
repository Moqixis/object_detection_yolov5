# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

# 导入thop包 用于计算FLOPs
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    """Detect模块是用来构建Detect层的，将输入feature map 通过一个卷积操作和公式计算到我们想要的shape, 为后面的计算损失或者NMS作准备"""
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        """
        detection layer 相当于yolov3中的YOLOLayer层
        :params nc: number of classes
        :params anchors: 传入3个feature map上的所有anchor的大小（P3、P4、P5）
        :params ch: [128, 256, 512] 3个输出feature map的channel
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers  Detect的个数 3
        self.na = len(anchors[0]) // 2  # number of anchors   每个feature map的anchor个数 3
        self.grid = [torch.zeros(1)] * self.nl  # init grid   {list: 3}  tensor([0.]) X 3
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        # 模型中需要保存的参数一般有两种：一种是反向传播需要被optimizer更新的，称为parameter; 另一种不要被更新称为buffer
        # buffer的参数更新是在forward中，而optim.step只能更新nn.parameter类型的参数
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # output conv 对每个输出的feature map都要调用一次conv1x1
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        """
        :return train: 一个tensor list 存放三个元素   [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                       分别是 [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
                inference: 0 [1, 19200+4800+1200, 25] = [bs, anchor_num*grid_w*grid_h, xywh+c+20classes]
                           1 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                             [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
        """
        z = []  # inference output
        for i in range(self.nl): # 对三个feature map分别进行处理
            x[i] = self.m[i](x[i])  # conv  
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # 调整tensor的形状
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # self.training属于父类nn.Module的一个变量
            # model.train()的调用self.training = True; model.eval()的调用self.training = False
            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                # sigmoid控制数值范围, 对于xywh都做了sigmoid
                y = x[i].sigmoid()
                # 选择直接inplace置换,或者再重新拼接输出, 这里是yolov5回归机制的核心代码
                if self.inplace:
                    # xy坐标回归预测: bx = 2σ(tx) - 0.5 + cx | by = 2σ(ty) - 0.5 + cy
                    # box center的x,y的预测被乘以2并减去了0.5，所以这里的值域从yolov3里的(0,1)开区间，变成了(-0.5,1.5)
                    # 从表面理解是yolov5可以跨半个格点预测了，这样可以提高对格点周围的bbox的召回.
                    # 还有一个好处就是也解决了yolov3中因为sigmoid开区间而导致中心无法到达边界处的问题
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                    # wh宽高回归预测: bw = pw(2σ(tw))^2 | bh = ph(2σ(th))^2
                    # 值域从基于anchor宽高的(0, +∞)变成了(0, 4), 预测的框范围更精准了,通过sigmoid约束让回归的框比例尺寸更为合理
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    # 不同的预测特征层尺度不同, 需要乘上不同的系数回归到原图大小
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    # 不同的预测特征层使用anchor大小不同, 预测目标尺寸不同, 需要乘上相对于特征点的anchor大小
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    # 重新concat拼接在一起
                    y = torch.cat((xy, wh, conf), 4)
                # z是一个tensor list 三个元素 分别是[1, 19200, 25] [1, 4800, 25] [1, 1200, 25]
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        """
        构造网格
        """
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        # arange(1,10)产生的是10-1=9 由1-9组成的1维度张量 ，类型int
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        # torch.meshgrid（）的功能是生成网格，可以用于生成坐标
        # yv: tensor([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], ...])
        # xv: tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], ...])
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        # 构建网格的特征点
        # torch.stack()在维度上连接（concatenate）若干个张量,tensor.expand()将张量广播到新的形状
        # (80,80)&(80,80) -> (80,80,2) -> (1,3,80,80,2) 复制3份
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        # 构建网格的anchor, 每一个特征图(h,w)都有3种不同尺度的anchor
        # tensor.view()方法可以调整tensor的形状，stride是为了复原每个预测特征层的anchor的绝对大小, 因为每一层尺度是不一样的
        # (3,2) -> (1,3,1,1,2) -> (1,3,80,80,2)
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Model(nn.Module):
    # YOLOv5 model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        """
        :params cfg:模型配置文件
        :params ch: input img channels 一般是3 RGB文件
        :params nc: number of classes 数据集的类别个数
        :anchors: 一般是None
        """
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name # cfg file name = yolov5s.yaml
            # 如果配置文件中有中文，打开时要加encoding参数
            with open(cfg, encoding='ascii', errors='ignore') as f:
                # model dict  取到配置文件中每条的信息（没有注释内容）
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        # 设置类别数 一般不执行, 因为nc=self.yaml['nc']恒成立
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        # 重写anchor，一般不执行, 因为传进来的anchors一般都是None
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # 创建网络模型
        # self.model: 初始化的整个网络模型(包括Detect层结构)
        # self.save: 所有层结构中from不等于-1的序号，并排好序  [4, 6, 10, 14, 17, 20, 23]
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        # default class names ['0', '1', '2',..., '19']
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        # 获取Detect模块的stride(相对输入图像的下采样率)和anchors在当前Detect输出的feature map的尺度
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # 计算三个feature map下采样的倍率  [8, 16, 32]
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            # 检查anchor顺序与stride顺序是否一致
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            # 求出相对当前feature map的anchor大小 如[10, 13]/8 -> [1.25, 1.625]
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)  # 调用torch_utils.py下initialize_weights初始化模型权重
        self.info()  # 打印模型信息
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        # augmented inference, None  scale和flip 上下flip/左右flip
        # 是否在测试时也使用数据增强  Test Time Augmentation(TTA)
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        """
        TTA Test Time Augmentation
        """
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        # 这里相当于对输入x进行3次不同参数的测试数据增强推理, 每次的推理结果都保存在列表y中
        for si, fi in zip(s, f):
            # scale_img缩放图片尺寸
            # 通过普通的双线性插值实现，根据ratio来控制图片的缩放比例，最后通过pad 0补齐到原图的尺寸
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            # _descale_pred将推理结果恢复到相对原图图片尺寸
            # 如果f=2,进行上下翻转; 如果f=3,进行左右翻转
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        # 把第一层的后面一部分的预测结果去掉, 也把最后一层的前面一部分的预测结果去掉
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """
        :params x: 输入图像
        :params profile: True 可以做一些性能评估
        :params feature_vis: True 可以做一些特征可视化
        :return train: 一个tensor list 存放三个元素   [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                       分别是 [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
                inference: 0 [1, 19200+4800+1200, 25] = [bs, anchor_num*grid_w*grid_h, xywh+c+20classes]
                           1 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                             [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
        """
        # y列表用来保存中间特征图; dt用来记录每个模块执行10次的平均时长
        # y: 存放着self.save=True的每一层的输出，因为后面的层结构concat等操作要用到
        # dt: 在profile中做性能评估时使用
        y, dt = [], []  # outputs
        # 对sequence模型进行遍历操作, 不断地对输入x进行处理, 中间结果需要保存的时候另外存储到列表y中
        for m in self.model:
            # 前向推理每一层结构   m.i=index   m.f=from   m.type=类名   m.np=number of params
            # 如果只是对前一个模块的输出进行操作, 则需要提取直接保存的中间特征图进行操作,
            # 一般是concat处理, 对当前层与之前曾进行一个concat再卷积; detect模块也需要提取3个特征层来处理
            if m.f != -1:  # if not from previous layer
                # 这里需要做4个concat操作和1个Detect操作
                # concat操作如m.f=[-1, 6] x就有两个元素,一个是上一层的输出,另一个是index=6的层的输出 再送到x=m(x)做concat操作
                # Detect操作m.f=[17, 20, 23] x有三个元素,分别存放第17层第20层第23层的输出 再送到x=m(x)做Detect的forward
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # 打印日志信息  FLOPs time等
            # profile参数打开会记录每个模块的平均执行10次的时长和flops用于分析模型的瓶颈, 提高模型的执行速度和降低显存占用
            if profile:
                self._profile_one_layer(m, x, dt)
            # 使用当前模块对特征图进行处理
            # 如果是concat模块: 则x是一个特征图列表, 则对其进行拼接处理, 再交给下一个卷积模块;
            # 如果是C3, Conv等普通的模块: 则x是单一特征图
            # 如果是detct模块: 则x是3个特征图的列表 (训练与推理返回的内容不一样)
            x = m(x)  # run
            # 存放着self.save的每一层的输出，因为后面需要用来作concat等操作要用到  不在self.save层的输出就为None
            # self.save: 把所有层结构中from不是-1的值记下并排序 [4, 6, 10, 14, 17, 20, 23]
            y.append(x if m.i in self.save else None)  # save output
            # 特征可视化 可以自己改动想要哪层的特征进行可视化
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        """用在上面的__init__函数上
        将推理结果恢复到原图图片尺寸  Test Time Augmentation(TTA)中用到
        de-scale predictions following augmented inference (inverse operation)
        :params p: 推理结果
        :params flips:
        :params scale:
        :params img_size:
        """
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:  # 默认执行 不使用AWS Inferentia
            p[..., :4] /= scale  # de-scale  xywh坐标缩放回原来大小
            # f=2,进行上下翻转
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            # f=3,进行左右翻转
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    # 这里y的一个包含3个子列表的列表, 通过对输入图像x进行了3次不同尺度的变换, 所以得到了3个inference结构
    # 这里看不太懂, 不过大概做的事情就是对第一个列表与最后一个列表的结果做一些过滤处理
    # 把第一层的后面一部分的预测结果去掉, 也把最后一层的前面一部分的预测结果去掉, 然后剩下的concat为一个部分
    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices  (25200 // 21) * 1 = 1200
        y[0] = y[0][:, :-i]  # large  (1,25200,25) -> (1,24000,25)
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices  (12348 // 21) * 16 = 9408
        y[-1] = y[-1][:, i:]  # small  (1,12348,25) -> (1,2940,25)
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        # profile函数返回flops与params, [0]表示计算浮点数
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        # 查看模块执行10次的平均时长
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        # 记录相关参数
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        # 到最后检测头的时候再输出总花费时长
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    # 对Detect()进行初始化
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    # 打印模型中最后Detect层的偏置bias信息(也可以任选哪些层bias信息)
    def _print_biases(self):
        """
        打印模型中最后Detect层的偏置bias信息(也可以任选哪些层bias信息)
        """
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    # 参数重结构化: 融合conv2d + batchnorm2d (推理的时候用, 可以加快模型的推理速度)
    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        """用在detect.py、val.py
        fuse model Conv2d() + BatchNorm2d() layers
        调用torch_utils.py中的fuse_conv_and_bn函数和common.py中Conv模块的fuseforward函数
        """
        LOGGER.info('Fusing layers... ')
        # 遍历每一层结构
        for m in self.model.modules():
            # 如果当前层是卷积层Conv与DWConv(继承Conv)且有bn结构, 那么就调用fuse_conv_and_bn函数讲conv和bn进行融合, 加速推理
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                # 融合Conv模块中的conv与bn层(不包含激活函数), 返回的是参数融合后的卷积
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                # 融合后conv的参数就包含了bn的用途, 所以可以删除bn层
                delattr(m, 'bn')  # remove batchnorm
                # 由于不需要bn层, 所以forward函数需要改写：
                # self.act(self.bn(self.conv(x))) -> self.act(self.conv(x))
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        """用在上面的__init__函数上
        调用torch_utils.py下model_info函数打印模型信息
        """
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch):  # model_dict, input_channels(3)
    """用在上面Model模块中
    解析模型文件(字典形式)，并搭建网络结构
    这个函数其实主要做的就是: 更新当前层的args（参数）,计算c2（当前层的输出channel） =>
                          使用当前层的参数搭建当前层 =>
                          生成 layers + save
    :params d: model_dict 模型文件 字典形式 {dict:7}  yolov5s.yaml中的6个元素 + ch
    :params ch: 记录模型每一层的输出channel 初始ch=[3] 后面会删除
    :return nn.Sequential(*layers): 网络的每一层的层结构
    :return sorted(save): 把所有层结构中from不是-1的值记下 并排序 [4, 6, 10, 14, 17, 20, 23]
    """
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # 读取d字典中的anchors和parameters(nc、depth_multiple、width_multiple)
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # na: number of anchors 每一个predict head上的anchor数 = 3
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # no: number of outputs 每一个predict head层的输出channel = anchors * (classes + 5) = 75(VOC)
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    # 开始搭建网络
    # layers: 保存每一层的层结构
    # save: 记录下所有层结构中from中不是-1的层结构序号
    # c2: 保存当前层的输出channel
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # from(当前层输入来自哪些层), number(当前层次数 初定), module(当前层类别), args(当前层类参数 初定)
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # eval(string) 得到当前层的真实类名 例如: m= Focus -> <class 'models.common.Focus'>
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass
        # ------------------- 更新当前层的args（参数）,计算c2（当前层的输出channel） -------------------
        # depth gain 控制深度  如v5s: n*0.33   n: 当前模块的次数(间接控制深度)
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in (Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost):
            # c1: 当前层的输入的channel数  c2: 当前层的输出的channel数(初定)  ch: 记录着所有层的输出channel
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                # width gain 控制宽度  如v5s: c2*0.5  c2: 当前层的最终输出的channel数(间接控制宽度)
                c2 = make_divisible(c2 * gw, 8)

            # 在初始arg的基础上更新 加入当前层的输入channel并更新当前层
            # [in_channel, out_channel, *args[1:]]
            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats  在第二个位置插入bottleneck个数n
                n = 1  # 恢复默认值1
        elif m is nn.BatchNorm2d:
            # BN层只需要返回上一层的输出channel
            args = [ch[f]]
        elif m is Concat:
            # Concat层则将f中所有的输出累加得到这层的输出channel
            c2 = sum(ch[x] for x in f)
        elif m is Detect:  # Detect（YOLO Layer）层
            # 在args中加入三个Detect层的输出channel
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        # -----------------------------------------------------------------------------------

        # m_: 得到当前层module  如果n>1就创建多个m(当前层结构), 如果n=1就创建一个m
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        # 打印当前层结构的一些基本信息
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        # append to savelist  把所有层结构中from不是-1的值记下  [6, 4, 14, 10, 17, 20, 23]
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        # 将当前层结构module加入layers中
        layers.append(m_)
        if i == 0:
            ch = []  # 去除输入channel [3]
        # 把当前层的输出channel数加入ch
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        _ = model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
