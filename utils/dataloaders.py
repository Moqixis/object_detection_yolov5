# YOLOv5 ð by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           cv2, segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html

# Get orientation exif tag
# ä¸é¨ä¸ºæ°ç ç¸æºçç§çèè®¾å®  å¯ä»¥è®°å½æ°ç ç§ççå±æ§ä¿¡æ¯åæææ°æ®
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

# è¿åæä»¶åè¡¨çhashå¼
def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash

# è·åæ°ç ç¸æºçå¾çå®½é«ä¿¡æ¯  å¹¶ä¸å¤æ­æ¯å¦éè¦æè½¬ï¼æ°ç ç¸æºå¯ä»¥å¤è§åº¦ææï¼
def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except Exception:
        pass

    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,}.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def create_dataloader(path,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      quad=False,
                      prefix='',
                      shuffle=False):
    """å¨train.pyä¸­è¢«è°ç¨ï¼ç¨äºçæTrainloader, datasetï¼testloader
    èªå®ä¹dataloaderå½æ°: è°ç¨LoadImagesAndLabelsè·åæ°æ®é(åæ¬æ°æ®å¢å¼º) + è°ç¨åå¸å¼éæ ·å¨DistributedSampler +
                        èªå®ä¹InfiniteDataLoader è¿è¡æ°¸ä¹æç»­çéæ ·æ°æ®
    :param path: å¾çæ°æ®å è½½è·¯å¾ train/test  å¦: ../datasets/VOC/images/train2007
    :param imgsz: train/testå¾çå°ºå¯¸ï¼æ°æ®å¢å¼ºåå¤§å°ï¼ 640
    :param batch_size: batch size å¤§å° 8/16/32
    :param stride: æ¨¡åæå¤§stride=32   [32 16 8]
    :param single_cls: æ°æ®éæ¯å¦æ¯åç±»å« é»è®¤False
    :param hyp: è¶ååè¡¨dict ç½ç»è®­ç»æ¶çä¸äºè¶åæ°ï¼åæ¬å­¦ä¹ çç­ï¼è¿éä¸»è¦ç¨å°éé¢ä¸äºå³äºæ°æ®å¢å¼º(æè½¬ãå¹³ç§»ç­)çç³»æ°
    :param augment: æ¯å¦è¦è¿è¡æ°æ®å¢å¼º  True
    :param cache: æ¯å¦cache_images False
    :param pad: è®¾ç½®ç©å½¢è®­ç»çshapeæ¶è¿è¡çå¡«å é»è®¤0.0
    :param rect: æ¯å¦å¼å¯ç©å½¢train/test  é»è®¤è®­ç»éå³é­ éªè¯éå¼å¯
    :param rank:  å¤å¡è®­ç»æ¶çè¿ç¨ç¼å· rankä¸ºè¿ç¨ç¼å·  -1ä¸gpu=1æ¶ä¸è¿è¡åå¸å¼  -1ä¸å¤ågpuä½¿ç¨DataParallelæ¨¡å¼  é»è®¤-1
    :param workers: dataloaderçnumworks å è½½æ°æ®æ¶çcpuè¿ç¨æ°
    :param image_weights: è®­ç»æ¶æ¯å¦æ ¹æ®å¾çæ ·æ¬çå®æ¡åå¸æéæ¥éæ©å¾ç  é»è®¤False
    :param quad: dataloaderåæ°æ®æ¶, æ¯å¦ä½¿ç¨collate_fn4ä»£æ¿collate_fn  é»è®¤False
    :param prefix: æ¾ç¤ºä¿¡æ¯   ä¸ä¸ªæ å¿ï¼å¤ä¸ºtrain/valï¼å¤çæ ç­¾æ¶ä¿å­cacheæä»¶ä¼ç¨å°
    """
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    # ä¸»è¿ç¨å®ç°æ°æ®çé¢è¯»åå¹¶ç¼å­ï¼ç¶åå¶å®å­è¿ç¨åä»ç¼å­ä¸­è¯»åæ°æ®å¹¶è¿è¡ä¸ç³»åè¿ç®ã
    # ä¸ºäºå®ææ°æ®çæ­£å¸¸åæ­¥, yolov5åºäºtorch.distributed.barrier()å½æ°å®ç°äºä¸ä¸æç®¡çå¨
    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        # è½½å¥æä»¶æ°æ®(å¢å¼ºæ°æ®é)
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    # åå¸å¼éæ ·å¨DistributedSampler
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    # ä½¿ç¨InfiniteDataLoaderå_RepeatSampleræ¥å¯¹DataLoaderè¿è¡å°è£, ä»£æ¿åDåçDataLoader, è½å¤æ°¸ä¹æç»­çéæ ·æ°æ®
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn), dataset


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    å½image_weights=Falseæ¶å°±ä¼è°ç¨è¿ä¸¤ä¸ªå½æ° è¿è¡èªå®ä¹DataLoader
    https://github.com/ultralytics/yolov5/pull/876
    ä½¿ç¨InfiniteDataLoaderå_RepeatSampleræ¥å¯¹DataLoaderè¿è¡å°è£, ä»£æ¿ååçDataLoader, è½å¤æ°¸ä¹æç»­çéæ ·æ°æ®
    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # è°ç¨_RepeatSamplerè¿è¡æç»­éæ ·
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever
    è¿é¨åæ¯è¿è¡æç»­éæ ·
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) # imgåæåå­è¿ç»­çæ°æ®  å å¿«è¿ç®

        # è¿åè·¯å¾, resize+padçå¾ç, åå§å¾ç, è§é¢å¯¹è±¡
        return path, img, img0, self.cap, s

    def new_video(self, path):
        # è®°å½å¸§æ°
        self.frame = 0
        # åå§åè§é¢å¯¹è±¡
        self.cap = cv2.VideoCapture(path)
        # å¾å°è§é¢æä»¶ä¸­çæ»å¸§æ°
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    """ç¨å°å¾å° load webç½é¡µä¸­çæ°æ®"""
    # YOLOv5 local webcam dataloader, i.e. `python detect.py --source 0`
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        s = f'webcam {self.count}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None, s

    def __len__(self):
        return 0


class LoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    """
    load æä»¶å¤¹ä¸­è§é¢æµ
    multiple IP or RTSP cameras
    å®ä¹è¿­ä»£å¨ ç¨äºdetect.py
    """
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream' # åå§åmodeä¸ºimages
        self.img_size = img_size
        self.stride = stride # æå¤§ä¸éæ ·æ­¥é¿

        # å¦æsourcesä¸ºä¸ä¸ªä¿å­äºå¤ä¸ªè§é¢æµçæä»¶  è·åæ¯ä¸ä¸ªè§é¢æµï¼ä¿å­ä¸ºä¸ä¸ªåè¡¨
        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            # åä¹ï¼åªæä¸ä¸ªè§é¢æµæä»¶å°±ç´æ¥ä¿å­
            sources = [sources]

        n = len(sources)  # è§é¢æµä¸ªæ°
        # åå§åå¾ç fps æ»å¸§æ° çº¿ç¨æ°
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        # éåæ¯ä¸ä¸ªè§é¢æµ
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            # å½åè§é¢index/æ»è§é¢æ°/è§é¢æµå°å
            st = f'{i + 1}/{n}: {s}... '
            if urlparse(s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl==2020.12.2'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            # s='0'æå¼æ¬å°æåå¤´ï¼å¦åæå¼è§é¢æµå°å
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            # è·åè§é¢çå®½åé¿
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # è·åè§é¢çå¸§æ°ï¼å¸§ç
            fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            # è¯»åå½åç»é¢
            _, self.imgs[i] = cap.read()  # guarantee first frame
            # åå»ºå¤çº¿ç¨è¯»åè§é¢æµï¼daemonè¡¨ç¤ºä¸»çº¿ç¨ç»ææ¶å­çº¿ç¨ä¹ç»æ
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            LOGGER.warning('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            # æ¯1å¸§è¯»åä¸æ¬¡  1->read
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

        # Stack  å°è¯»åçå¾çæ¼æ¥å°ä¸èµ·
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'JPEGImages' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 prefix=''):
        """
        åå§åè¿ç¨å¹¶æ²¡æä»ä¹å®è´¨æ§çæä½,æ´å¤æ¯ä¸ä¸ªå®ä¹åæ°çè¿ç¨ï¼selfåæ°ï¼,ä»¥ä¾¿å¨__getitem()__ä¸­è¿è¡æ°æ®å¢å¼ºæä½,æä»¥è¿é¨åä»£ç åªéè¦æä½selfä¸­çåä¸ªåéçå«ä¹å°±ç®å·®ä¸å¤äº
        self.img_files: {list: N} å­æ¾çæ´ä¸ªæ°æ®éå¾ççç¸å¯¹è·¯å¾
        self.label_files: {list: N} å­æ¾çæ´ä¸ªæ°æ®éå¾ççç¸å¯¹è·¯å¾
        cache label -> verify_image_label
        self.labels: å¦ææ°æ®éææå¾çä¸­æ²¡æä¸ä¸ªå¤è¾¹å½¢label  labelså­å¨çlabelå°±é½æ¯åå§label(é½æ¯æ­£å¸¸çç©å½¢label)
                     å¦åå°ææå¾çæ­£å¸¸gtçlabelå­å¥labels ä¸æ­£å¸¸gt(å­å¨ä¸ä¸ªå¤è¾¹å½¢)ç»è¿segments2boxesè½¬æ¢ä¸ºæ­£å¸¸çç©å½¢label
        self.shapes: ææå¾ççshape
        self.segments: å¦ææ°æ®éææå¾çä¸­æ²¡æä¸ä¸ªå¤è¾¹å½¢label  self.segments=None
                       å¦åå­å¨æ°æ®éä¸­ææå­å¨å¤è¾¹å½¢gtçå¾ççææåå§label(è¯å®æå¤è¾¹å½¢label ä¹å¯è½æç©å½¢æ­£å¸¸label æªç¥æ°)
        self.batch: è®°è½½çæ¯å¼ å¾çå±äºåªä¸ªbatch
        self.n: æ°æ®éä¸­ææå¾ççæ°é
        self.indices: è®°è½½çææå¾ççindex
        self.rect=Trueæ¶self.batch_shapesè®°è½½æ¯ä¸ªbatchçshape(åä¸ä¸ªbatchçå¾çshapeç¸å)
        """
        # 1ãèµå¼ä¸äºåºç¡çselfåé ç¨äºåé¢å¨__getitem__ä¸­è°ç¨
        self.img_size = img_size    # ç»è¿æ°æ®å¢å¼ºåçæ°æ®å¾ççå¤§å°
        self.augment = augment      # æ¯å¦å¯å¨æ°æ®å¢å¼º ä¸è¬è®­ç»æ¶æå¼ éªè¯æ¶å³é­
        self.hyp = hyp              # è¶ååè¡¨
        # å¾çææééæ ·  Trueå°±å¯ä»¥æ ¹æ®ç±»å«é¢ç(é¢çé«çæéå°,åæ­£å¤§)æ¥è¿è¡éæ ·  é»è®¤False: ä¸ä½ç±»å«åºå
        self.image_weights = image_weights
        self.rect = False if image_weights else rect  # æ¯å¦å¯å¨ç©å½¢è®­ç» ä¸è¬è®­ç»æ¶å³é­ éªè¯æ¶æå¼ å¯ä»¥å é
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        # mosaicå¢å¼ºçè¾¹çå¼  [-320, -320]
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride    # æå¤§ä¸éæ ·ç 32
        self.path = path        # å¾çè·¯å¾
        self.albumentations = Albumentations() if augment else None

        # 2ãå¾å°pathè·¯å¾ä¸çææå¾ççè·¯å¾self.img_files  è¿ééè¦èªå·±debugä¸ä¸ ä¸ä¼å¤ªé¾
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                # è·åæ°æ®éè·¯å¾pathï¼åå«å¾çè·¯å¾çtxtæä»¶æèåå«å¾ççæä»¶å¤¹è·¯å¾
                # ä½¿ç¨pathlib.Pathçæä¸æä½ç³»ç»æ å³çè·¯å¾ï¼å ä¸ºä¸åæä½ç³»ç»è·¯å¾çâ/âä¼ææä¸å
                p = Path(p)  # os-agnostic
                # å¦æè·¯å¾pathä¸ºåå«å¾ççæä»¶å¤¹è·¯å¾
                if p.is_dir():  # dir
                    # glob.glab: è¿åææå¹éçæä»¶è·¯å¾åè¡¨  éå½è·åpè·¯å¾ä¸æææä»¶
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                # å¦æè·¯å¾pathä¸ºåå«å¾çè·¯å¾çtxtæä»¶
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        # è·åæ°æ®éè·¯å¾çä¸çº§ç¶ç®å½  os.sepä¸ºè·¯å¾éçåéç¬¦ï¼ä¸åè·¯å¾çåéç¬¦ä¸åï¼os.sepå¯ä»¥æ ¹æ®ç³»ç»èªéåºï¼
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            # ç ´æå·æ¿æ¢ä¸ºos.sepï¼os.path.splitext(x)å°æä»¶åä¸æ©å±ååå¼å¹¶è¿åä¸ä¸ªåè¡¨
            # ç­éfä¸­ææçå¾çæä»¶
            self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.im_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')

        # Check cache
        # 3ãæ ¹æ®imgsè·¯å¾æ¾å°labelsçè·¯å¾self.label_files
        self.label_files = img2label_paths(self.im_files)  # labels
        # 4ãcache label ä¸æ¬¡è¿è¡è¿ä¸ªèæ¬çæ¶åç´æ¥ä»cacheä¸­ålabelèä¸æ¯å»æä»¶ä¸­ålabel éåº¦æ´å¿«
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            # å¦ææcacheæä»¶ï¼ç´æ¥å è½½  exists=True: æ¯å¦å·²ä»cacheæä»¶ä¸­è¯»åºäºnf, nm, ne, nc, nç­ä¿¡æ¯
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # same version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # same hash
        except Exception:
            # å¦åè°ç¨cache_labelsç¼å­æ ç­¾åæ ç­¾ç¸å³ä¿¡æ¯
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        # æ°æ®éæ²¡ææ ç­¾ä¿¡æ¯ å°±ååºè­¦åå¹¶æ¾ç¤ºæ ç­¾labelä¸è½½å°åhelp_url
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'

        # Read cache
        # 5ãRead cache  ä»cacheä¸­è¯»åºææ°åéèµç»self  æ¹ä¾¿ç»forwardä¸­ä½¿ç¨
        # cacheä¸­çé®å¼å¯¹æåæ: cache[img_file]=[l, shape, segments] cache[hash] cache[results] cache[msg] cache[version]
        # åä»cacheä¸­å»é¤cacheæä»¶ä¸­å¶ä»æ å³é®å¼å¦:'hash', 'version', 'msgs'ç­é½å é¤
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        # popæresultsãhashãversionãmsgsååªå©ä¸cache[img_file]=[l, shape, segments]
        # cache.values(): åcacheä¸­ææå¼ å¯¹åºææl, shape, segments
        # labels: å¦ææ°æ®éææå¾çä¸­æ²¡æä¸ä¸ªå¤è¾¹å½¢label  labelså­å¨çlabelå°±é½æ¯åå§label(é½æ¯æ­£å¸¸çç©å½¢label)
        #         å¦åå°ææå¾çæ­£å¸¸gtçlabelå­å¥labels ä¸æ­£å¸¸gt(å­å¨ä¸ä¸ªå¤è¾¹å½¢)ç»è¿segments2boxesè½¬æ¢ä¸ºæ­£å¸¸çç©å½¢label
        # shapes: ææå¾ççshape
        # self.segments: å¦ææ°æ®éææå¾çä¸­æ²¡æä¸ä¸ªå¤è¾¹å½¢label  self.segments=None
        #                å¦åå­å¨æ°æ®éä¸­ææå­å¨å¤è¾¹å½¢gtçå¾ççææåå§label(è¯å®æå¤è¾¹å½¢label ä¹å¯è½æç©å½¢æ­£å¸¸label æªç¥æ°)
        # zip æ¯å ä¸ºcacheä¸­æælabelsãshapesãsegmentsä¿¡æ¯é½æ¯ææ¯å¼ imgåå¼å­å¨ç, zipæ¯å°ææå¾çå¯¹åºçä¿¡æ¯å å¨ä¸èµ·
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.im_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n       # number of images
        self.indices = range(n) # ææå¾ççindex

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0

        # Rectangular Training
        # 6ãä¸ºRectangular Trainingä½åå¤
        # è¿éä¸»è¦æ¯æ³¨æshapesççæ è¿ä¸æ­¥å¾éè¦ å ä¸ºå¦æéæ ·ç©å½¢è®­ç»é£ä¹æ´ä¸ªbatchçå½¢ç¶è¦ä¸æ · å°±è¦è®¡ç®è¿ä¸ªç¬¦åæ´ä¸ªbatchçshape
        # èä¸è¿è¦å¯¹æ°æ®éæç§é«å®½æ¯è¿è¡æåº è¿æ ·æè½ä¿è¯åä¸ä¸ªbatchçå¾ççå½¢ç¶å·®ä¸å¤ç¸å åéåä¸ä¸ªå±åçshapeä»£ä»·ä¹æ¯è¾å°
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()    # æ ¹æ®é«å®½æ¯æåº
            self.im_files = [self.im_files[i] for i in irect]           # è·åæåºåçimg_files
            self.label_files = [self.label_files[i] for i in irect]     # è·åæåºåçlabel_files
            self.labels = [self.labels[i] for i in irect]               # è·åæåºåçlabels
            self.shapes = s[irect]  # wh                                # è·åæåºåçwh
            ar = ar[irect]                                              # è·åæåºåçaspect ratio

            # Set training image shapes
            # è®¡ç®æ¯ä¸ªbatchéç¨çç»ä¸å°ºåº¦ Set training image shapes
            shapes = [[1, 1]] * nb  # nb: number of batches
            for i in range(nb):
                ari = ar[bi == i]   # bi: batch index
                mini, maxi = ari.min(), ari.max()   # è·åç¬¬iä¸ªbatchä¸­ï¼æå°åæå¤§é«å®½æ¯
                # å¦æé«/å®½å°äº1(w > h)ï¼å°wè®¾ä¸ºimg_sizeï¼ä¿è¯åå¾åå°ºåº¦ä¸åè¿è¡ç¼©æ¾ï¼
                if maxi < 1:
                    shapes[i] = [maxi, 1]   # maxi: hç¸å¯¹æå®å°ºåº¦çæ¯ä¾  1: wç¸å¯¹æå®å°ºåº¦çæ¯ä¾
                # å¦æé«/å®½å¤§äº1(w < h)ï¼å°hè®¾ç½®ä¸ºimg_sizeï¼ä¿è¯åå¾åå°ºåº¦ä¸åè¿è¡ç¼©æ¾ï¼
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            # è®¡ç®æ¯ä¸ªbatchè¾å¥ç½ç»çshapeå¼(åä¸è®¾ç½®ä¸º32çæ´æ°å)
            # è¦æ±æ¯ä¸ªbatch_shapesçé«å®½é½æ¯32çæ´æ°åï¼æä»¥è¦åé¤ä»¥32ï¼åæ´åä¹ä»¥32ï¼ä¸è¿img_sizeå¦ææ¯32åæ°è¿éå°±æ²¡å¿è¦äºï¼
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # 7ãæ¯å¦éè¦cache image ä¸è¬æ¯False å ä¸ºRAMä¼ä¸è¶³  cache labelè¿å¯ä»¥ ä½æ¯cache imageå°±å¤ªå¤§äº æä»¥ä¸è¬ä¸ç¨
        # Cache images into RAM/disk for faster training (WARNING: large datasets may exceed system resources)
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == 'disk' else self.load_image
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            pbar = tqdm(enumerate(results), total=n, bar_format=BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache_images == 'disk':
                    gb += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.ims[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        """ç¨å¨__init__å½æ°ä¸­  cacheæ°æ®élabel
        å è½½labelä¿¡æ¯çæcacheæä»¶   Cache dataset labels, check images and read shapes
        :params path: cacheæä»¶ä¿å­å°å
        :params prefix: æ¥å¿å¤´é¨ä¿¡æ¯(å½©æé«äº®é¨å)
        :return x: cacheä¸­ä¿å­çå­å¸
               åæ¬çä¿¡æ¯æ: x[im_file] = [l, shape, segments]
                          ä¸å¼ å¾çä¸ä¸ªlabelç¸å¯¹åºçä¿å­å°x, æç»xä¼ä¿å­ææå¾ççç¸å¯¹è·¯å¾ãgtæ¡çä¿¡æ¯ãå½¢ç¶shapeãææçå¤è¾¹å½¢gtä¿¡æ¯
                              im_file: å½åè¿å¼ å¾ççpathç¸å¯¹è·¯å¾
                              l: å½åè¿å¼ å¾ççæægtæ¡çlabelä¿¡æ¯(ä¸åå«segmentå¤è¾¹å½¢æ ç­¾) [gt_num, cls+xywh(normalized)]
                              shape: å½åè¿å¼ å¾ççå½¢ç¶ shape
                              segments: å½åè¿å¼ å¾çæægtçlabelä¿¡æ¯(åå«segmentå¤è¾¹å½¢æ ç­¾) [gt_num, xy1...]
                           hash: å½åå¾çålabelæä»¶çhashå¼  1
                           results: æ¾å°çlabelä¸ªæ°nf, ä¸¢å¤±labelä¸ªæ°nm, ç©ºlabelä¸ªæ°ne, ç ´ælabelä¸ªæ°nc, æ»img/labelä¸ªæ°len(self.img_files)
                           msgs: æææ°æ®éçmsgsä¿¡æ¯
                           version: å½åcache version
        """
        x = {}  # åå§åæç»cacheä¸­ä¿å­çå­å¸dict
        # åå§ånumber missing, found, empty, corrupt, messages
        # åå§åæ´ä¸ªæ°æ®é: æ¼æçæ ç­¾(label)æ»æ°é, æ¾å°çæ ç­¾(label)æ»æ°é, ç©ºçæ ç­¾(label)æ»æ°é, éè¯¯æ ç­¾(label)æ»æ°é, ææéè¯¯ä¿¡æ¯
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."  # æ¥å¿
        # å¤è¿ç¨è°ç¨verify_image_labelå½æ°
        with Pool(NUM_THREADS) as pool:
            # å®ä¹pbarè¿åº¦æ¡
            # pool.imap_unordered: å¯¹å¤§éæ°æ®éåå¤è¿ç¨è®¡ç® è¿åä¸ä¸ªè¿­ä»£å¨
            # æself.img_files, self.label_files, repeat(prefix) listä¸­çå¼ä½ä¸ºåæ°ä¾æ¬¡éå¥(ä¸æ¬¡éä¸ä¸ª)verify_image_labelå½æ°
            pbar = tqdm(pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
                        desc=desc,
                        total=len(self.im_files),
                        bar_format=BAR_FORMAT)
            # im_file: å½åè¿å¼ å¾ççpathç¸å¯¹è·¯å¾
            # l: [gt_num, cls+xywh(normalized)]
            #    å¦æè¿å¼ å¾çæ²¡æä¸ä¸ªsegmentå¤è¾¹å½¢æ ç­¾ lå°±å­å¨ålabel(å¨é¨æ¯æ­£å¸¸ç©å½¢æ ç­¾)
            #    å¦æè¿å¼ å¾çæä¸ä¸ªsegmentå¤è¾¹å½¢æ ç­¾  lå°±å­å¨ç»è¿segments2boxeså¤çå¥½çæ ç­¾(æ­£å¸¸ç©å½¢æ ç­¾ä¸å¤ç å¤è¾¹å½¢æ ç­¾è½¬åä¸ºç©å½¢æ ç­¾)
            # shape: å½åè¿å¼ å¾ççå½¢ç¶ shape
            # segments: å¦æè¿å¼ å¾çæ²¡æä¸ä¸ªsegmentå¤è¾¹å½¢æ ç­¾ å­å¨None
            #           å¦æè¿å¼ å¾çæä¸ä¸ªsegmentå¤è¾¹å½¢æ ç­¾ å°±æè¿å¼ å¾ççæælabelå­å¨å°segmentsä¸­(è¥å¹²ä¸ªæ­£å¸¸gt è¥å¹²ä¸ªå¤è¾¹å½¢æ ç­¾) [gt_num, xy1...]
            # nm_f(nm): number missing å½åè¿å¼ å¾ççlabelæ¯å¦ä¸¢å¤±         ä¸¢å¤±=1    å­å¨=0
            # nf_f(nf): number found å½åè¿å¼ å¾ççlabelæ¯å¦å­å¨           å­å¨=1    ä¸¢å¤±=0
            # ne_f(ne): number empty å½åè¿å¼ å¾ççlabelæ¯å¦æ¯ç©ºç         ç©ºç=1    æ²¡ç©º=0
            # nc_f(nc): number corrupt å½åè¿å¼ å¾ççlabelæä»¶æ¯å¦æ¯ç ´æç  ç ´æç=1  æ²¡ç ´æ=0
            # msg: è¿åçmsgä¿¡æ¯  labelæä»¶å®å¥½=ââ  labelæä»¶ç ´æ=warningä¿¡æ¯
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f  # ç´¯å æ»number missing label
                nf += nf_f  # ç´¯å æ»number found label
                ne += ne_f  # ç´¯å æ»number empty label
                nc += nc_f  # ç´¯å æ»number corrupt label
                if im_file:
                    x[im_file] = [lb, shape, segments]  # ä¿¡æ¯å­å¥å­å¸ key=im_file
                if msg:
                    msgs.append(msg)  # å°msgå å¥æ»msg
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"  # æ¥å¿

        pbar.close()# å³é­è¿åº¦æ¡
        # æ¥å¿æå°ææmsgä¿¡æ¯
        if msgs:
            LOGGER.info('\n'.join(msgs))
        # ä¸å¼ labelé½æ²¡æ¾å° æ¥å¿æå°help_urlä¸è½½å°å
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files) # å°å½åå¾çålabelæä»¶çhashå¼å­å¥æç»å­å¸dist
        x['results'] = nf, nm, ne, nc, len(self.im_files) # å°nf, nm, ne, nc, len(self.img_files)å­å¥æç»å­å¸dist
        x['msgs'] = msgs  # warnings  å°æææ°æ®éçmsgsä¿¡æ¯å­å¥æç»å­å¸dist
        x['version'] = self.cache_version  # cache version  å°å½åcache versionå­å¥æç»å­å¸dist
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    def __len__(self):
        return len(self.im_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        """
        è¿é¨åæ¯æ°æ®å¢å¼ºå½æ°ï¼ä¸è¬ä¸æ¬¡æ§æ§è¡batch_sizeæ¬¡ã
        è®­ç» æ°æ®å¢å¼º: mosaic(random_perspective) + hsv + ä¸ä¸å·¦å³ç¿»è½¬
        æµè¯ æ°æ®å¢å¼º: letterbox
        :return torch.from_numpy(img): è¿ä¸ªindexçå¾çæ°æ®(å¢å¼ºå) [3, 640, 640]
        :return labels_out: è¿ä¸ªindexå¾ççgt label [6, 6] = [gt_num, 0+class+xywh(normalized)]
        :return self.img_files[index]: è¿ä¸ªindexå¾ççè·¯å¾å°å
        :return shapes: è¿ä¸ªbatchçå¾ççshapes æµè¯æ¶(ç©å½¢è®­ç»)ææ  éªè¯æ¶ä¸ºNone   for COCO mAP rescaling
        """
        # è¿éå¯ä»¥éè¿ä¸ç§å½¢å¼è·åè¦è¿è¡æ°æ®å¢å¼ºçå¾çindex  linear, shuffled, or image_weights
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp # è¶å åå«ä¼å¤æ°æ®å¢å¼ºè¶å
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        # mosaicå¢å¼º å¯¹å¾åè¿è¡4å¼ å¾æ¼æ¥è®­ç»  ä¸è¬è®­ç»æ¶è¿è¡
        # mosaic + MixUp
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            # mixupæ°æ®å¢å¼º
            if random.random() < hyp['mixup']:
                # *load_mosaic(self, random.randint(0, self.n - 1)) éæºä»æ°æ®éä¸­ä»»éä¸å¼ å¾çåæ¬å¼ å¾çè¿è¡mixupæ°æ®å¢å¼º
                # img:   ä¸¤å¼ å¾çèåä¹åçå¾ç numpy (640, 640, 3)
                # labels: ä¸¤å¼ å¾çèåä¹åçæ ç­¾label [M+N, cls+x1y1x2y2]
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

                # æµè¯ä»£ç  æµè¯MixUpææ
                # cv2.imshow("MixUp", img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # print(img.shape)   # (640, 640, 3)
        
        # å¦å: è½½å¥å¾ç + Letterbox  (val)
        else:
            # Load image
            # è½½å¥å¾ç  è½½å¥å¾çåè¿ä¼è¿è¡ä¸æ¬¡resize  å°å½åå¾ççæé¿è¾¹ç¼©æ¾å°æå®çå¤§å°(512), è¾å°è¾¹åæ¯ä¾ç¼©æ¾
            # load image img=(343, 512, 3)=(h, w, c)  (h0, w0)=(335, 500)  numpy  index=4
            # img: resizeåçå¾ç   (h0, w0): åå§å¾ççhw  (h, w): resizeåçå¾ççhw
            # è¿ä¸æ­¥æ¯å°(335, 500, 3) resize-> (343, 512, 3)
            img, (h0, w0), (h, w) = self.load_image(index)

            # æµè¯ä»£ç  æµè¯load_imageææ
            # cv2.imshow("load_image", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print(img.shape)   # (640, 640, 3)

            # Letterbox
            # letterboxä¹åç¡®å®è¿å¼ å½åå¾çletterboxä¹åçshape  å¦æä¸ç¨self.rectç©å½¢è®­ç»shapeå°±æ¯self.img_size
            # å¦æä½¿ç¨self.rectç©å½¢è®­ç»shapeå°±æ¯å½åbatchçshape å ä¸ºç©å½¢è®­ç»çè¯æä»¬æ´ä¸ªbatchçshapeå¿é¡»ç»ä¸(å¨__init__å½æ°ç¬¬6èåå®¹)
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            # letterbox è¿ä¸æ­¥å°ç¬¬ä¸æ­¥ç¼©æ¾å¾å°çå¾çåç¼©æ¾å°å½åbatchæéè¦çå°ºåº¦ (343, 512, 3) pad-> (384, 512, 3)
            # (ç©å½¢æ¨çéè¦ä¸ä¸ªbatchçææå¾ççshapeå¿é¡»ç¸åï¼èè¿ä¸ªshapeå¨initå½æ°ä¸­ä¿æå¨self.batch_shapesä¸­)
            # è¿éæ²¡æç¼©æ¾æä½ï¼æä»¥è¿éçratioæ°¸è¿é½æ¯(1.0, 1.0)  pad=(0.0, 20.5)
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # å¾çletterboxä¹ålabelçåæ ä¹è¦ç¸åºåå  æ ¹æ®padè°æ´labelåæ  å¹¶å°å½ä¸åçxywh -> æªå½ä¸åçxyxy
            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            # æµè¯ä»£ç  æµè¯letterboxææ
            # cv2.imshow("letterbox", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print(img.shape)   # (640, 640, 3)

            if self.augment:
                # ä¸åmosaicçè¯å°±è¦årandom_perspectiveå¢å¼º å ä¸ºmosaicå½æ°åé¨æ§è¡äºrandom_perspectiveå¢å¼º
                # random_perspectiveå¢å¼º: éæºå¯¹å¾çè¿è¡æè½¬ï¼å¹³ç§»ï¼ç¼©æ¾ï¼è£åªï¼éè§åæ¢
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            # xyxy to xywh normalized
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        # å¹³ç§»å¢å¼º éæºå·¦å³ç¿»è½¬ + éæºä¸ä¸ç¿»è½¬
        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            # è²åç©ºé´å¢å¼ºAugment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # æµè¯ä»£ç  æµè¯augment_hsvææ
            # cv2.imshow("augment_hsv", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print(img.shape)   # (640, 640, 3)

            # éæºä¸ä¸ç¿»è½¬ flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)  # np.flipud å°æ°ç»å¨ä¸ä¸æ¹åç¿»è½¬
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]  # 1 - y_center  labelä¹è¦æ å°

            # éæºå·¦å³ç¿»è½¬ Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)  # np.fliplr å°æ°ç»å¨å·¦å³æ¹åç¿»è½¬
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]   # 1 - x_center  labelä¹è¦æ å°

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        # 6ä¸ªå¼çtensor åå§åæ ç­¾æ¡å¯¹åºçå¾çåºå·, éåä¸é¢çcollate_fnä½¿ç¨
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i],
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if self.augment else cv2.INTER_AREA  # random.choice(self.rand_interp_methods)
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        else:
            return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_images_to_disk(self, i):
        # Saves an image as an *.npy file for faster loading
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        """ç¨å¨LoadImagesAndLabelsæ¨¡åç__getitem__å½æ° è¿è¡mosaicæ°æ®å¢å¼º
        å°åå¼ å¾çæ¼æ¥å¨ä¸å¼ é©¬èµåå¾åä¸­  loads images in a 4-mosaic
        :param index: éè¦è·åçå¾åç´¢å¼
        :return: img4: mosaicåéæºéè§åæ¢åçä¸å¼ å¾ç  numpy(640, 640, 3)
                labels4: img4å¯¹åºçtarget  [M, cls+x1y1x2y2]
        """
        # labels4: ç¨äºå­æ¾æ¼æ¥å¾åï¼4å¼ å¾æ¼æä¸å¼ ï¼çlabelä¿¡æ¯(ä¸åå«segmentså¤è¾¹å½¢)
        # segments4: ç¨äºå­æ¾æ¼æ¥å¾åï¼4å¼ å¾æ¼æä¸å¼ ï¼çlabelä¿¡æ¯(åå«segmentså¤è¾¹å½¢)
        labels4, segments4 = [], []
        s = self.img_size # ä¸è¬çå¾çå¤§å°
        # éæºåå§åæ¼æ¥å¾åçä¸­å¿ç¹åæ   [0, s*2]ä¹é´éæºå2ä¸ªæ°ä½ä¸ºæ¼æ¥å¾åçä¸­å¿åæ 
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        # ä»datasetä¸­éæºå¯»æ¾é¢å¤çä¸å¼ å¾åè¿è¡æ¼æ¥ [14, 26, 2, 16] åéæºéä¸å¼ å¾ççindex
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        # éååå¼ å¾åè¿è¡æ¼æ¥ 4å¼ ä¸åå¤§å°çå¾å => 1å¼ [1472, 1472, 3]çå¾å
        for i, index in enumerate(indices):
            # Load image  æ¯æ¬¡æ¿ä¸å¼ å¾ç å¹¶å°è¿å¼ å¾çresizeå°self.size(h,w)
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left  åå¾[375, 500, 3] load_image->[552, 736, 3]   hwc
                # åå»ºé©¬èµåå¾å [1472, 1472, 3]=[h, w, c]
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                # è®¡ç®é©¬èµåå¾åä¸­çåæ ä¿¡æ¯(å°å¾åå¡«åå°é©¬èµåå¾åä¸­)   w=736  h = 552  é©¬èµåå¾åï¼(x1a,y1a)å·¦ä¸è§ (x2a,y2a)å³ä¸è§
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                # è®¡ç®æªåçå¾ååºåä¿¡æ¯(ä»¥xc,ycä¸ºç¬¬ä¸å¼ å¾åçå³ä¸è§åæ å¡«åå°é©¬èµåå¾åä¸­ï¼ä¸¢å¼è¶ççåºå)  å¾åï¼(x1b,y1b)å·¦ä¸è§ (x2b,y2b)å³ä¸è§
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                # è®¡ç®æªåçå¾ååºåä¿¡æ¯(ä»¥xc,ycä¸ºç¬¬äºå¼ å¾åçå·¦ä¸è§åæ å¡«åå°é©¬èµåå¾åä¸­ï¼ä¸¢å¼è¶ççåºå)
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                # è®¡ç®æªåçå¾ååºåä¿¡æ¯(ä»¥xc,ycä¸ºç¬¬ä¸å¼ å¾åçå³ä¸è§åæ å¡«åå°é©¬èµåå¾åä¸­ï¼ä¸¢å¼è¶ççåºå)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                # è®¡ç®æªåçå¾ååºåä¿¡æ¯(ä»¥xc,ycä¸ºç¬¬åå¼ å¾åçå·¦ä¸è§åæ å¡«åå°é©¬èµåå¾åä¸­ï¼ä¸¢å¼è¶ççåºå)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            # å°æªåçå¾ååºåå¡«åå°é©¬èµåå¾åçç¸åºä½ç½®   img4[h, w, c]
            # å°å¾åimgçã(x1b,y1b)å·¦ä¸è§ (x2b,y2b)å³ä¸è§ãåºåæªååºæ¥å¡«åå°é©¬èµåå¾åçã(x1a,y1a)å·¦ä¸è§ (x2a,y2a)å³ä¸è§ãåºå
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            # è®¡ç®pad(å½åå¾åè¾¹çä¸é©¬èµåè¾¹ççè·ç¦»ï¼è¶ççæåµpadw/padhä¸ºè´å¼)  ç¨äºåé¢çlabelæ å°
            padw = x1a - x1b  # å½åå¾åä¸é©¬èµåå¾åå¨wç»´åº¦ä¸ç¸å·®å¤å°
            padh = y1a - y1b  # å½åå¾åä¸é©¬èµåå¾åå¨hç»´åº¦ä¸ç¸å·®å¤å°

            # Labels
            # labels: è·åå¯¹åºæ¼æ¥å¾åçæææ­£å¸¸labelä¿¡æ¯(å¦ææsegmentså¤è¾¹å½¢ä¼è¢«è½¬åä¸ºç©å½¢label)
            # segments: è·åå¯¹åºæ¼æ¥å¾åçææä¸æ­£å¸¸labelä¿¡æ¯(åå«segmentså¤è¾¹å½¢ä¹åå«æ­£å¸¸gt)
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                # normalized xywh normalized to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)      # æ´æ°labels4
            segments4.extend(segments)  # æ´æ°segments4

        # Concat/clip labels  ælabels4ï¼[(2, 5), (1, 5), (3, 5), (1, 5)] => (7, 5)ï¼åç¼©å°ä¸èµ·
        labels4 = np.concatenate(labels4, 0)
        # é²æ­¢è¶ç  label[:, 1:]ä¸­çææåç´ çå¼ï¼ä½ç½®ä¿¡æ¯ï¼å¿é¡»å¨[0, 2*s]ä¹é´,å°äº0å°±ä»¤å¶ç­äº0,å¤§äº2*så°±ç­äº2*s   out: è¿å
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # æµè¯ä»£ç   æµè¯åé¢çmosaicææ
        # cv2.imshow("mosaic", img4)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(img4.shape)   # (1280, 1280, 3)

        # éæºåç§»æ ç­¾ä¸­å¿ï¼çææ°çæ ç­¾ä¸åæ ç­¾ç»å replicate
        # img4, labels4 = replicate(img4, labels4)
        #
        # # æµè¯ä»£ç   æµè¯replicateææ
        # cv2.imshow("replicate", img4)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(img4.shape)   # (1280, 1280, 3)

        # Augment
        # random_perspective Augment  éæºéè§åæ¢ [1280, 1280, 3] => [640, 640, 3]
        # å¯¹mosaicæ´ååçå¾çè¿è¡éæºæè½¬ãå¹³ç§»ãç¼©æ¾ãè£åªï¼éè§åæ¢ï¼å¹¶resizeä¸ºè¾å¥å¤§å°img_size
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
        img4, labels4 = random_perspective(img4,
                                           labels4,
                                           segments4,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        # æµè¯ä»£ç  æµè¯mosaic + random_perspectiveéæºä»¿å°åæ¢ææ
        # cv2.imshow("random_perspective", img4)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(img4.shape)   # (640, 640, 3)
        return img4, labels4

    def load_mosaic9(self, index):
        # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
        """ç¨å¨LoadImagesAndLabelsæ¨¡åç__getitem__å½æ° æ¿æ¢mosaicæ°æ®å¢å¼º
        å°ä¹å¼ å¾çæ¼æ¥å¨ä¸å¼ é©¬èµåå¾åä¸­  loads images in a 9-mosaic
        :param self:
        :param index: éè¦è·åçå¾åç´¢å¼
        :return: img9: mosaicåä»¿å°å¢å¼ºåçä¸å¼ å¾ç
                labels9: img9å¯¹åºçtarget
        """
        # labels9: ç¨äºå­æ¾æ¼æ¥å¾åï¼9å¼ å¾æ¼æä¸å¼ ï¼çlabelä¿¡æ¯(ä¸åå«segmentså¤è¾¹å½¢)
        # segments9: ç¨äºå­æ¾æ¼æ¥å¾åï¼9å¼ å¾æ¼æä¸å¼ ï¼çlabelä¿¡æ¯(åå«segmentså¤è¾¹å½¢)
        labels9, segments9 = [], []
        s = self.img_size  # ä¸è¬çå¾çå¤§å°(ä¹æ¯æç»è¾åºçå¾çå¤§å°)
        # ä»datasetä¸­éæºå¯»æ¾é¢å¤çä¸å¼ å¾åè¿è¡æ¼æ¥ [14, 26, 2, 16] åéæºéä¸å¼ å¾ççindex
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        random.shuffle(indices)
        hp, wp = -1, -1  # height, width previous
        for i, index in enumerate(indices):
            # Load image   æ¯æ¬¡æ¿ä¸å¼ å¾ç å¹¶å°è¿å¼ å¾çresizeå°self.size(h,w)
            img, _, (h, w) = self.load_image(index)

            # è¿éåä¸é¢load_mosaicå½æ°çæä½ç±»ä¼¼ å°±æ¯å°ååºçimgå¾çåµå°img9ä¸­(ä¸æ¯ççåµå¥ èæ¯æ¾å°å¯¹åºçä½ç½®)
            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # åä¸é¢load_mosaicå½æ°çæä½ç±»ä¼¼ æ¾å°mosaic9å¢å¼ºåçlabels9åsegments9
            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # çæå¯¹åºçimg9å¾ç(å°å¯¹åºä½ç½®çå¾çåµå¥img9ä¸­)
            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc
        c = np.array([xc, yc])  # centers
        segments9 = [x - c for x in segments9]

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img9, labels9 = replicate(img9, labels9)  # replicate

        # Augment åæ ·è¿è¡ éæºéè§åæ¢
        img9, labels9 = random_perspective(img9,
                                           labels9,
                                           segments9,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img9, labels9

    @staticmethod
    def collate_fn(batch):
        """è¿ä¸ªå½æ°ä¼å¨create_dataloaderä¸­çædataloaderæ¶è°ç¨ï¼
        æ´çå½æ°  å°imageålabelæ´åå°ä¸èµ·
        :return torch.stack(img, 0): å¦[16, 3, 640, 640] æ´ä¸ªbatchçå¾ç
        :return torch.cat(label, 0): å¦[15, 6] [num_target, img_index+class_index+xywh(normalized)] æ´ä¸ªbatchçlabel
        :return path: æ´ä¸ªbatchææå¾ççè·¯å¾
        :return shapes: (h0, w0), ((h / h0, w / w0), pad)    for COCO mAP rescaling
        pytorchçDataLoaderæåä¸ä¸ªbatchçæ°æ®éæ¶è¦ç»è¿æ­¤å½æ°è¿è¡æå éè¿éåæ­¤å½æ°å®ç°æ ç­¾ä¸å¾çå¯¹åºçååï¼ä¸ä¸ªbatchä¸­åªäºæ ç­¾å±äºåªä¸å¼ å¾ç,å½¢å¦
            [[0, 6, 0.5, 0.5, 0.26, 0.35],
             [0, 6, 0.5, 0.5, 0.26, 0.35],
             [1, 6, 0.5, 0.5, 0.26, 0.35],
             [2, 6, 0.5, 0.5, 0.26, 0.35],]
           åä¸¤è¡æ ç­¾å±äºç¬¬ä¸å¼ å¾ç, ç¬¬ä¸è¡å±äºç¬¬äºå¼ ããã
        """
        # img: ä¸ä¸ªtuple ç±batch_sizeä¸ªtensorç»æ æ´ä¸ªbatchä¸­æ¯ä¸ªtensorè¡¨ç¤ºä¸å¼ å¾ç
        # label: ä¸ä¸ªtuple ç±batch_sizeä¸ªtensorç»æ æ¯ä¸ªtensorå­æ¾ä¸å¼ å¾ççææçtargetä¿¡æ¯
        #        label[6, object_num] 6ä¸­çç¬¬ä¸ä¸ªæ°ä»£è¡¨ä¸ä¸ªbatchä¸­çç¬¬å å¼ å¾
        # path: ä¸ä¸ªtuple ç±4ä¸ªstrç»æ, æ¯ä¸ªstrå¯¹åºä¸å¼ å¾ççå°åä¿¡æ¯
        im, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        # è¿åçimg=[batch_size, 3, 736, 736]
        #      torch.stack(img, 0): å°batch_sizeä¸ª[3, 736, 736]çç©éµæ¼æä¸ä¸ª[batch_size, 3, 736, 736]
        # label=[target_sums, 6]  6ï¼è¡¨ç¤ºå½åtargetå±äºåªä¸å¼ å¾+class+x+y+w+h
        #      torch.cat(label, 0): å°[n1,6]ã[n2,6]ã[n3,6]...æ¼æ¥æ[n1+n2+n3+..., 6]
        # è¿éä¹æä»¥æ¼æ¥çæ¹å¼ä¸åæ¯å ä¸ºimgæ¼æ¥çæ¶åå®çæ¯ä¸ªé¨åçå½¢ç¶æ¯ç¸åçï¼é½æ¯[3, 736, 736]
        # èælabelçæ¯ä¸ªé¨åçå½¢ç¶æ¯ä¸ä¸å®ç¸åçï¼æ¯å¼ å¾çç®æ ä¸ªæ°æ¯ä¸ä¸å®ç¸åçï¼labelè¯å®ä¹å¸æç¨stack,æ´æ¹ä¾¿,ä½æ¯ä¸è½é£æ ·æ¼ï¼
        # å¦ææ¯å¼ å¾çç®æ ä¸ªæ°æ¯ç¸åçï¼é£æä»¬å°±å¯è½ä¸éè¦éåcollate_fnå½æ°äº
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        """åæ ·å¨create_dataloaderä¸­çædataloaderæ¶è°ç¨ï¼
        è¿éæ¯yolo-v5ä½èå®éªæ§çä¸ä¸ªä»£ç  quad-collate function å½train.pyçoptåæ°quad=True åè°ç¨collate_fn4ä»£æ¿collate_fn
        ä½ç¨:  å¦ä¹åç¨collate_fnå¯ä»¥è¿åå¾ç[16, 3, 640, 640] ç»è¿collate_fn4åè¿åå¾ç[4, 3, 1280, 1280]
              å°4å¼ mosaicå¾ç[1, 3, 640, 640]åæä¸å¼ å¤§çmosaicå¾ç[1, 3, 1280, 1280]
              å°ä¸ä¸ªbatchçå¾çæ¯åå¼ å¤ç, 0.5çæ¦çå°åå¼ å¾çæ¼æ¥å°ä¸å¼ å¤§å¾ä¸è®­ç», 0.5æ¦çç´æ¥å°æå¼ å¾çä¸éæ ·ä¸¤åè®­ç»
        """
        # img: æ´ä¸ªbatchçå¾ç [16, 3, 640, 640]
        # label: æ´ä¸ªbatchçlabelæ ç­¾ [num_target, img_index+class_index+xywh(normalized)]
        # path: æ´ä¸ªbatchææå¾ççè·¯å¾
        # shapes: (h0, w0), ((h / h0, w / w0), pad)    for COCO mAP rescaling
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4  # collate_fn4å¤çåè¿ä¸ªbatchä¸­å¾ççä¸ªæ°
        im4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]  # åå§å

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4  # éæ · [0, 4, 8, 16]
            # éæºæ°å°äº0.5å°±ç´æ¥å°æå¼ å¾çä¸éæ ·ä¸¤åè®­ç»
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear',
                                   align_corners=False)[0].type(img[i].type())
                lb = label[i]
            # éæºæ°å¤§äº0.5å°±å°åå¼ å¾ç(mosaicåç)æ¼æ¥å°ä¸å¼ å¤§å¾ä¸è®­ç»
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            im4.append(im)
            label4.append(lb)

        # åé¢è¿åçé¨ååcollate_fnå°±å·®ä¸å¤äº åå åè§£éé½åå¨ä¸ä¸ä¸ªå½æ°äº èªå·±debugçä¸ä¸å§
        for i, lb in enumerate(label4):
            lb[:, 0] = i  # add target image index for build_targets()

        return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4

# ä¸¤ä¸ªæä»¶æä½
# Ancillary functions --------------------------------------------------------------------------------------------------
def create_folder(path='./new'):
    # Create folder
    """ç¨å¨flatten_recursiveå½æ°ä¸­ ä½flatten_recursiveå½æ°æ²¡ç¨å° æä»¥è¿ä¸ªå½æ°ä¹æ²¡ç¨å°
    åå»ºæä»¶å¤¹  Create folder
    """
    # å¦æpathå­å¨æä»¶å¤¹ï¼åç§»é¤
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    # åä»æ°æ°å»ºè¿ä¸ªæä»¶å¤¹
    os.makedirs(path)  # make new output folder


def flatten_recursive(path=DATASETS_DIR / 'coco128'):
    # Flatten a recursive directory by bringing all files to top level
    """æ²¡ç¨å°  ä¸æ¯å¾éè¦
    å°ä¸ä¸ªæä»¶è·¯å¾ä¸­çæææä»¶å¤å¶å°å¦ä¸ä¸ªæä»¶å¤¹ä¸­  å³å°imageæä»¶ålabelæä»¶æ¾å°ä¸ä¸ªæ°æä»¶å¤¹ä¸­
    Flatten a recursive directory by bringing all files to top level
    """
    new_path = Path(str(path) + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        # shutil.copyfile: å¤å¶æä»¶å°å¦ä¸ä¸ªæä»¶å¤¹ä¸­
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path=DATASETS_DIR / 'coco128'):  # from utils.datasets import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    """èªè¡ä½¿ç¨ çæåç±»æ°æ®é
    å°ç®æ æ£æµæ°æ®éè½¬åä¸ºåç±»æ°æ®é éä½åæ³: æç®æ æ£æµæ°æ®éä¸­çæ¯ä¸ä¸ªgtæè§£å¼ åç±»å«å­å¨å°å¯¹åºçæä»¶å½ä¸­
    Convert detection dataset into classification dataset, with one directory per class
    ä½¿ç¨: from utils.datasets import *; extract_boxes()
    :params path: æ°æ®éå°å
    """
    path = Path(path)  # images dir  æ°æ®éæä»¶ç®å½ é»è®¤'..\datasets\coco128'
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))  # éå½éåpathæä»¶ä¸ç'*.*'æä»¶
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS: # å¿é¡»å¾æ¯å¾çæä»¶
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2] # å¾å°è¿å¼ å¾çh w

            # labels æ ¹æ®è¿å¼ å¾ççè·¯å¾æ¾å°è¿å¼ å¾ççlabelè·¯å¾
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file) as f:
                    # è¯»ålabelçåè¡: å¯¹åºåä¸ªgtåæ 
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):  # éåæ¯ä¸ä¸ªgt
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    # f.parent: 'F:\yolo_v5\datasets\coco128\images\train2017\classifier\45'
                    if not f.parent.is_dir():
                        # æ¯ä¸ä¸ªç±»å«çç¬¬ä¸å¼ ç§çå­è¿å»ä¹å ååå»ºå¯¹åºç±»çæä»¶å¤¹
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box  normalized to æ­£å¸¸å¤§å°
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int) # xywh to xyxy

                    # é²æ­¢båºç clip boxes outside of image
                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'

# èªè¡ä½¿ç¨ èªè¡ååæ°æ®é
def autosplit(path=DATASETS_DIR / 'coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    # è·åimagesä¸­ææçå¾ç image files only
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility   éæºæ°ç§å­
    # assign each image to a split æ ¹æ®(train, val, test)æéåååå§å¾çæ°æ®é
    # indices: [n]   0, 1, 2   åå«è¡¨ç¤ºæ°æ®éä¸­æ¯ä¸å¼ å¾çå±äºåªä¸ªæ°æ®é åå«å¯¹åºç(train, val, test)
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path.parent / x).unlink(missing_ok=True) for x in txt]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')  # add image to txt file


def verify_image_label(args):
    """ç¨å¨cache_labelså½æ°ä¸­
    æ£æµæ°æ®éä¸­æ¯å¼ å¾çåæ¯å¼ laeblæ¯å¦å®å¥½
    å¾çæä»¶: åå®¹ãæ ¼å¼ãå¤§å°ãå®æ´æ§
    labelæä»¶: æ¯ä¸ªgtå¿é¡»æ¯ç©å½¢(æ¯è¡é½å¾æ¯5ä¸ªæ° class+xywh) + æ ç­¾æ¯å¦å¨é¨>=0 + æ ç­¾åæ xywhæ¯å¦å½ä¸å + æ ç­¾ä¸­æ¯å¦æéå¤çåæ 
    :params im_file: æ°æ®éä¸­ä¸å¼ å¾ççpathç¸å¯¹è·¯å¾
    :params lb_file: æ°æ®éä¸­ä¸å¼ å¾ççlabelç¸å¯¹è·¯å¾
    :params prefix: æ¥å¿å¤´é¨ä¿¡æ¯(å½©æé«äº®é¨å)
    :return im_file: å½åè¿å¼ å¾ççpathç¸å¯¹è·¯å¾
    :return l: [gt_num, cls+xywh(normalized)]
               å¦æè¿å¼ å¾çæ²¡æä¸ä¸ªsegmentå¤è¾¹å½¢æ ç­¾ lå°±å­å¨ålabel(å¨é¨æ¯æ­£å¸¸ç©å½¢æ ç­¾)
               å¦æè¿å¼ å¾çæä¸ä¸ªsegmentå¤è¾¹å½¢æ ç­¾  lå°±å­å¨ç»è¿segments2boxeså¤çå¥½çæ ç­¾(æ­£å¸¸ç©å½¢æ ç­¾ä¸å¤ç å¤è¾¹å½¢æ ç­¾è½¬åä¸ºç©å½¢æ ç­¾)
    :return shape: å½åè¿å¼ å¾ççå½¢ç¶ shape
    :return segments: å¦æè¿å¼ å¾çæ²¡æä¸ä¸ªsegmentå¤è¾¹å½¢æ ç­¾ å­å¨None
                      å¦æè¿å¼ å¾çæä¸ä¸ªsegmentå¤è¾¹å½¢æ ç­¾ å°±æè¿å¼ å¾ççæælabelå­å¨å°segmentsä¸­(è¥å¹²ä¸ªæ­£å¸¸gt è¥å¹²ä¸ªå¤è¾¹å½¢æ ç­¾) [gt_num, xy1...]
    :return nm: number missing å½åè¿å¼ å¾ççlabelæ¯å¦ä¸¢å¤±         ä¸¢å¤±=1    å­å¨=0
    :return nf: number found å½åè¿å¼ å¾ççlabelæ¯å¦å­å¨           å­å¨=1    ä¸¢å¤±=0
    :return ne: number empty å½åè¿å¼ å¾ççlabelæ¯å¦æ¯ç©ºç         ç©ºç=1    æ²¡ç©º=0
    :return nc: number corrupt å½åè¿å¼ å¾ççlabelæä»¶æ¯å¦æ¯ç ´æç  ç ´æç=1  æ²¡ç ´æ=0
    :return msg: è¿åçmsgä¿¡æ¯  labelæä»¶å®å¥½=ââ  labelæä»¶ç ´æ=warningä¿¡æ¯
    """
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # æ£æ¥è¿å¼ å¾ç(åå®¹ãæ ¼å¼ãå¤§å°ãå®æ´æ§) verify images
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify  æ£æ¥å¾çåå®¹åæ ¼å¼æ¯å¦æ­£å¸¸
        shape = exif_size(im)  # image size  # å½åå¾ççå¤§å°
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'  # å¾çå¤§å°å¿é¡»å¤§äº9ä¸ªpixels
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'  # å¾çæ ¼å¼å¿é¡»å¨img_formatä¸­
        if im.format.lower() in ('jpg', 'jpeg'):  # æ£æ¥jpgæ ¼å¼æä»¶
            with open(im_file, 'rb') as f:
                # f.seek: -2 åç§»é åæä»¶å¤´æ¹åä¸­ç§»å¨çå­èæ°   2 ç¸å¯¹ä½ç½® ä»æä»¶å°¾å¼å§åç§»
                f.seek(-2, 2)
                # f.read(): è¯»åå¾çæä»¶  æä»¤: \xff\xd9  æ£æµæ´å¼ å¾çæ¯å¦å®æ´  å¦æä¸å®æ´å°±è¿åcorrupted JPEG
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file): # å¦æè¿ä¸ªlabelè·¯å¾å­å¨
            nf = 1  # label found
            with open(lb_file) as f:  # è¯»ålabelæä»¶
                # è¯»åå½ålabelæä»¶çæ¯ä¸è¡: æ¯ä¸è¡é½æ¯å½åå¾ççä¸ä¸ªgt
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                # any() å½æ°ç¨äºå¤æ­ç»å®çå¯è¿­ä»£åæ° æ¯å¦å¨é¨ä¸ºFalse,åè¿å False; å¦ææä¸ä¸ªä¸º True,åè¿åTrue
                # å¦æå½åå¾ççlabelæä»¶æä¸åæ°å¤§äº8, åè®¤ä¸ºlabelæ¯å­å¨segmentçpolygonç¹(å¤è¾¹å½¢)  å°±ä¸æ¯ç©éµ åå°labelä¿¡æ¯å­å¥segmentä¸­
                if any(len(x) > 6 for x in lb):  # is segment
                    # å½åå¾çä¸­æægtæ¡çç±»å«
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    # è·å¾è¿å¼ å¾ä¸­æægtæ¡çlabelä¿¡æ¯(åå«segmentå¤è¾¹å½¢æ ç­¾)
                    # å ä¸ºsegmentæ ç­¾å¯ä»¥æ¯ä¸åé¿åº¦ï¼æä»¥è¿ésegmentsæ¯ä¸ä¸ªåè¡¨ [gt_num, xy1...(normalized)]
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    # è·å¾è¿å¼ å¾ä¸­æægtæ¡çlabelä¿¡æ¯(ä¸åå«segmentå¤è¾¹å½¢æ ç­¾)
                    # segments(å¤è¾¹å½¢) -> bbox(æ­£æ¹å½¢), å¾å°æ°æ ç­¾  [gt_num, cls+xywh(normalized)]
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)  # l: to float32
            nl = len(lb)
            if nl:
                # å¤æ­æ ç­¾æ¯å¦å¨é¨>=0
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                # å¤æ­æ ç­¾åæ x y w hæ¯å¦å½ä¸å
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
                # å¤æ­æ ç­¾ä¸­æ¯å¦æéå¤çåæ 
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = segments[i]
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty l.shape[0] == 0åä¸ºç©ºçæ ç­¾ï¼ne=1
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing  ä¸å­å¨æ ç­¾æä»¶ï¼ånm = 1
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False, profile=False, hub=False):
    """ Return dataset statistics dictionary with images and instances counts per split per class
    yolov5æ°æ®éæ²¡æç¨  èªè¡ä½¿ç¨ 
    è¿åæ°æ®éçç¶æå­å¸  åå«: æ¯ä¸ªç±»å«çå¾çæ°é  æ¯ä¸ªç±»å«çå®ä¾æ°é
    To run in parent directory: export PYTHONPATH="$PWD/yolov5"
    Usage1: from utils.datasets import *; dataset_stats('coco128.yaml', autodownload=True)
    Usage2: from utils.datasets import *; dataset_stats('path/to/coco128_with_yaml.zip')
    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
        verbose:        Print stats dictionary
    """

    def round_labels(labels):
        # Update labels to integer class and 6 decimal place floats
        return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

    def unzip(path):
        # Unzip data.zip TODO: CONSTRAINT: path/to/abc.zip MUST unzip to 'path/to/abc/'
        if str(path).endswith('.zip'):  # path is data.zip
            assert Path(path).is_file(), f'Error unzipping {path}, file not found'
            ZipFile(path).extractall(path=path.parent)  # unzip
            dir = path.with_suffix('')  # dataset directory == zip name
            return True, str(dir), next(dir.rglob('*.yaml'))  # zipped, data_dir, yaml_path
        else:  # path is data.yaml
            return False, None, path

    def hub_ops(f, max_dim=1920):
        # HUB ops for 1 image 'f': resize and save at reduced quality in /dataset-hub for web/app viewing
        f_new = im_dir / Path(f).name  # dataset-hub image filename
        try:  # use PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # ratio
            if r < 1.0:  # image too large
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, 'JPEG', quality=75, optimize=True)  # save
        except Exception as e:  # use OpenCV
            print(f'WARNING: HUB ops PIL failure {f}: {e}')
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)  # ratio
            if r < 1.0:  # image too large
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(f_new), im)

    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_yaml(yaml_path), errors='ignore') as f:
        data = yaml.safe_load(f)  # data dict
        if zipped:
            data['path'] = data_dir  # TODO: should this be dir.resolve()?
    # æ£æ¥æ°æ®éæ¯å¦å­å¨ download dataset if missing
    check_dataset(data, autodownload)  # download dataset if missing
    hub_dir = Path(data['path'] + ('-hub' if hub else ''))
    stats = {'nc': data['nc'], 'names': data['names']}  # statistics dictionary
    for split in 'train', 'val', 'test':  # åtrainãvalãtestç»è®¡æ°æ®éä¿¡æ¯
        if data.get(split) is None:
            stats[split] = None  # i.e. no test set
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])  # load dataset
        # ç»è®¡æ°æ®éæ¯ä¸ªå¾çlabelä¸­æ¯ä¸ªç±»å«gtæ¡çä¸ªæ°
        # x: {list: img_num} æ¯ä¸ªlist[class_num]  æ¯ä¸ªå¾ççlabelä¸­æ¯ä¸ªç±»å«gtæ¡çä¸ªæ°
        for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data['nc']))
        # list to numpyç©éµ  [img_num, class_num]
        x = np.array(x)  # shape(128x80)
        # åå«ç»è®¡trainãvalãtestä¸ä¸ªæ°æ®éçæ°æ®ä¿¡æ¯
        # åæ¬: 'image_stats': å­å¸dict  å¾çæ°étotal  æ²¡ææ ç­¾çæä»¶ä¸ªæ°unlabelled  æ°æ®éæ¯ä¸ªç±»å«çgtä¸ªæ°[80]
        # 'instance_stats': å­å¸dict  æ°æ®éä¸­ææå¾ççæægtä¸ªæ°total   æ°æ®éä¸­æ¯ä¸ªç±»å«çgtä¸ªæ°[80]
        # 'labels': å­å¸dict  key=æ°æ®éä¸­æ¯å¼ å¾ççæä»¶å  value=æ¯å¼ å¾çå¯¹åºçlabelä¿¡æ¯ [n, cls+xywh]
        stats[split] = {
            'instance_stats': {
                'total': int(x.sum()),
                'per_class': x.sum(0).tolist()},
            'image_stats': {
                'total': dataset.n,
                'unlabelled': int(np.all(x == 0, 1).sum()),
                'per_class': (x > 0).sum(0).tolist()},
            'labels': [{
                str(Path(k).name): round_labels(v.tolist())} for k, v in zip(dataset.im_files, dataset.labels)]}

        if hub:
            im_dir = hub_dir / 'images'
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(hub_ops, dataset.im_files), total=dataset.n, desc='HUB Ops'):
                pass

    # Profile
    stats_path = hub_dir / 'stats.json'
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix('.npy')
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(f'stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

            file = stats_path.with_suffix('.json')
            t1 = time.time()
            with open(file, 'w') as f:
                json.dump(stats, f)  # save stats *.json
            t2 = time.time()
            with open(file) as f:
                x = json.load(f)  # load hyps dict
            print(f'stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

    # Save, print and return  ç»è®¡ä¿¡æ¯stats
    if hub:
        print(f'Saving {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(stats, f)  # save stats.json
    if verbose:  # printå¯è§å
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats
