### 1.参考

☑YOLOv5超详细的入门级教程（训练篇）（二）——VOC2007模型复现 [🔗](https://blog.csdn.net/weixin_44145782/article/details/114042061)  [🔗](https://blog.csdn.net/qq_36756866/article/details/109111065)

使用yolov5训练自己voc格式的数据集，完整记录 [🔗](https://blog.csdn.net/weixin_48780159/article/details/119461483)



YOLOv5核心基础知识讲解 [🔗](https://blog.csdn.net/weixin_44227733/article/details/123489648)   [知乎](https://zhuanlan.zhihu.com/p/172121380)

YOLOv5：项目目录结构（五）[🔗](https://wenku.baidu.com/view/787bf4e2b84cf7ec4afe04a1b0717fd5360cb23e.html) [🔗](https://blog.csdn.net/weixin_44227733/article/details/123518494)      [others](https://blog.csdn.net/weixin_44227733/category_11624411.html)

YOLOv5代码详解(train->test->yolov5.yaml->common) [1](https://blog.csdn.net/mary_0830/article/details/107076617) [2](https://blog.csdn.net/mary_0830/article/details/107081030) [3](https://blog.csdn.net/mary_0830/article/details/107124459) [4](https://blog.csdn.net/mary_0830/article/details/107125686)

YOLOV5训练代码train.py注释与解析 [🔗](https://blog.csdn.net/Q1u1NG/article/details/107463417)   test.py  [🔗](https://blog.csdn.net/Q1u1NG/article/details/107464724)

yolov5 test.py val.py detec.py 区别在哪里呢?  [🔗](https://blog.csdn.net/orDream/article/details/122095535)

☑YOLOV5源码的详细解读   [🔗](https://blog.csdn.net/BGMcat/article/details/120930016)

### 2.FPS计算

作者给了回答，见👉https://github.com/ultralytics/yolov5/discussions/6713

### 3.超大数据集上训练停在epoch0不动的问题

~把img改成416，batchsize改成1，workers改成0，能改的都往小里改改看~
多GPU分布式数据并行训练[🔗](https://blog.csdn.net/u011922698/article/details/123244505): python -m torch.distributed.launch --nproc_per_node 2 train.py

### 4.视频检测假阳性过多问题

后处理：连续五帧内没有阳性则认为第五帧是假阳，直接输出原图
跳帧（实现了再补充）
