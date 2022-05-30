# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
General utils
"""
# YOLOv5 通用工具类代码

import contextlib
import glob
import inspect
import logging
import math
import os
import platform
import random
import re
import shutil
import signal
import time
import urllib
from datetime import datetime
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from typing import Optional
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml

from utils.downloads import gsutil_getsize
from utils.metrics import box_iou, fitness

# Settings
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
DATASETS_DIR = ROOT.parent / 'datasets'  # YOLOv5 datasets directory
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
AUTOINSTALL = str(os.getenv('YOLOv5_AUTOINSTALL', True)).lower() == 'true'  # global auto-install mode
VERBOSE = str(os.getenv('YOLOv5_VERBOSE', True)).lower() == 'true'  # global verbose mode
FONT = 'Arial.ttf'  # https://ultralytics.com/assets/Arial.ttf

# 设置运行相关的一些基本的配置  Settings
# 控制print打印torch.tensor格式设置  tensor精度为5(小数点后5位)  每行字符数为320个  显示方法为long
torch.set_printoptions(linewidth=320, precision=5, profile='long')
# 控制print打印np.array格式设置  精度为5  每行字符数为320个  format short g, %precision=5
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
# pandas的最大显示行数是10
pd.options.display.max_columns = 10
# 阻止opencv参与多线程(与 Pytorch的 Dataloader不兼容)
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
# 确定最大的线程数 这里被限制在了8
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  # NumExpr max threads
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)  # OpenMP max threads (PyTorch and SciPy)


def is_kaggle():
    # Is environment a Kaggle Notebook?
    try:
        assert os.environ.get('PWD') == '/kaggle/working'
        assert os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'
        return True
    except AssertionError:
        return False


def is_writeable(dir, test=False):
    # Return True if directory has write permissions, test opening a file with write permissions if test=True
    if test:  # method 1
        file = Path(dir) / 'tmp.txt'
        try:
            with open(file, 'w'):  # open file with write permissions
                pass
            file.unlink()  # remove file
            return True
        except OSError:
            return False
    else:  # method 2
        return os.access(dir, os.R_OK)  # possible issues on Windows


def set_logging(name=None, verbose=VERBOSE):
    """广泛使用在train.py、val.py、detect.py等文件的main函数的第一步
    对日志的设置(format、level)等进行初始化
    """
    # Sets level and returns logger
    if is_kaggle():
        for h in logging.root.handlers:
            logging.root.removeHandler(h)  # remove all handlers associated with the root logger object
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    # 设置日志级别  rank不为-1或0时设置输出级别level为WARN  为-1或0时设置级别为INFO
    level = logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    log.addHandler(handler)


set_logging()  # run before defining LOGGER
LOGGER = logging.getLogger("yolov5")  # define globally (used in train.py, val.py, detect.py, etc.)


def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
    # Return path of user configuration directory. Prefer environment variable if exists. Make dir if required.
    env = os.getenv(env_var)
    if env:
        path = Path(env)  # use environment variable
    else:
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}  # 3 OS dirs
        path = Path.home() / cfg.get(platform.system(), '')  # OS-specific config dir
        path = (path if is_writeable(path) else Path('/tmp')) / dir  # GCP and AWS lambda fix, only /tmp is writeable
    path.mkdir(exist_ok=True)  # make if required
    return path


CONFIG_DIR = user_config_dir()  # Ultralytics settings dir


class Profile(contextlib.ContextDecorator):
    # Usage: @Profile() decorator or 'with Profile():' context manager
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        print(f'Profile results: {time.time() - self.start:.5f}s')


class Timeout(contextlib.ContextDecorator):
    # Usage: @Timeout(seconds) decorator or 'with Timeout(seconds):' context manager
    """代码中都是使用库函数自己定义的timeout 没用用这个自定义的timeout函数
    设置一个超时函数 如果某个程序执行超时  就会触发超时处理函数_timeout_handler 返回超时异常信息
    并没有用到  这里面的timeout都是用python库函数实现的 并不需要自己另外写一个
    使用: with timeout(seconds):  sleep(10)   或者   @timeout(seconds) decorator
    dealing with wandb login-options timeout issues as well as check_github() timeout issues
    """
    def __init__(self, seconds, *, timeout_msg='', suppress_timeout_errors=True):
        self.seconds = int(seconds)         # 限制时间
        self.timeout_message = timeout_msg  # 报错信息
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        # 超时处理函数 一旦超时 就在seconds后发送超时信息
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        if platform.system() != 'Windows':  # not supported on Windows
            # signal.signal: 设置信号处理的函数_timeout_handler
            # 执行流进入with中会执行__enter__方法 如果发生超时, 就会触发超时处理函数_timeout_handler 返回超时异常信息
            signal.signal(signal.SIGALRM, self._timeout_handler)  # Set handler for SIGALRM
            # signal.alarm: 设置发送SIGALRM信号的定时器
            signal.alarm(self.seconds)  # start countdown for SIGALRM to be raised

    def __exit__(self, exc_type, exc_val, exc_tb):
        if platform.system() != 'Windows':
            # 执行流离开 with 块时(没有发生超时), 则调用这个上下文管理器的__exit__方法来清理所使用的资源
            signal.alarm(0)  # Cancel SIGALRM if it's scheduled
            if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
                return True


class WorkingDirectory(contextlib.ContextDecorator):
    # Usage: @WorkingDirectory(dir) decorator or 'with WorkingDirectory(dir):' context manager
    def __init__(self, new_dir):
        self.dir = new_dir  # new dir
        self.cwd = Path.cwd().resolve()  # current dir

    def __enter__(self):
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.cwd)


def try_except(func):
    # try-except function. Usage: @try_except decorator
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return handler


def methods(instance):
    # Get class/instance methods
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]


def print_args(args: Optional[dict] = None, show_file=True, show_fcn=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, fcn, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    s = (f'{Path(file).stem}: ' if show_file else '') + (f'{fcn}: ' if show_fcn else '')
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))


def init_seeds(seed=0):
    """在train函数的一开始调用
    用于设置一系列的随机数种子
    """
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    # 设置随机数 针对使用random.random()生成随机数的时候相同
    random.seed(seed)
    # 设置随机数 针对使用np.random.rand()生成随机数的时候相同
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def is_docker():
    # Is environment a Docker container?
    return Path('/workspace').exists()  # or Path('/.dockerenv').exists()


def is_colab():
    # Is environment a Google Colab instance?
    try:
        import google.colab
        return True
    except ImportError:
        return False


def is_pip():
    # Is file in a pip package?
    return 'site-packages' in Path(__file__).resolve().parts


def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)


def is_chinese(s='人工智能'):
    # Is string composed of any Chinese characters?
    return True if re.search('[\u4e00-\u9fff]', str(s)) else False


def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


def file_age(path=__file__):
    # Return days since last file update
    dt = (datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime))  # delta
    return dt.days  # + dt.seconds / 86400  # fractional days


def file_update_date(path=__file__):
    # Return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def file_size(path):
    # Return file/dir size (MB)
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0


def check_online():
    # Check internet connectivity
    """在下面的check_git_status、check_requirements等函数中使用
    检查当前主机网络连接是否可用
    """
    import socket  # 导入socket模块 可解决基于tcp和ucp协议的网络传输

    try:
        # 连接到一个ip 地址addr("1.1.1.1")的TCP服务上, 端口号port=443 timeout=5 时限5秒 并返回一个新的套接字对象
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        # 没发现什么异常, 连接成功, 有网, 就返回True
        return True
    except OSError:
        return False


def git_describe(path=ROOT):  # path must be a directory
    """用在select_device
    用于返回path文件可读的git描述  return human-readable git description  i.e. v5.0-5-g3e25f1e
    https://git-scm.com/docs/git-describe
    path: 需要在git中查询（文件描述）的文件名  默认当前文件的父路径
    """
    # Return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    try:
        assert (Path(path) / '.git').is_dir()
        return check_output(f'git -C {path} describe --tags --long --always', shell=True).decode()[:-1]
    except Exception:
        return ''


@try_except
@WorkingDirectory(ROOT)
def check_git_status():
    """用在train.py的main函数的一开始
    检查当前代码版本是否是最新的   如果不是最新的 会提示使用git pull命令进行升级
    """
    # Recommend 'git pull' if code is out of date
    msg = ', for updates see https://github.com/ultralytics/yolov5'
    s = colorstr('github: ')  # string
    # 检查电脑有没有安装git仓库  没有安装直接报异常并输出异常信息
    assert Path('.git').exists(), s + 'skipping check (not a git repository)' + msg
    # 检查电脑系统有没有安装docker环境变量 没有直接报异常并输出异常信息
    assert not is_docker(), s + 'skipping check (Docker image)' + msg
    # 检查主机是否联网
    assert check_online(), s + 'skipping check (offline)' + msg

    # 创建cmd命令
    cmd = 'git fetch && git config --get remote.origin.url'
    # 并创建子进程进行执行cmd命令  返回执行结果  时限5秒
    url = check_output(cmd, shell=True, timeout=5).decode().strip().rstrip('.git')  # git fetch
    branch = check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()  # checked out
    n = int(check_output(f'git rev-list {branch}..origin/master --count', shell=True))  # commits behind
    # n>0 说明当前版本之后还有commit 因此当前版本不是最新的 s为输出的相关提示
    if n > 0:
        # 如果不是最新  提升字符s: WARNING...
        s += f"⚠️ YOLOv5 is out of date by {n} commit{'s' * (n > 1)}. Use `git pull` or `git clone {url}` to update."
    else:
        # 已经是最新
        s += f'up to date with {url} ✅'
    LOGGER.info(emojis(s))  # emoji-safe


def check_python(minimum='3.7.0'):
    """用在下面的函数check_requirements中
    检查当前的版本号是否满足最小版本号minimum
    Check current python version vs. required python version
    """
    # Check current python version vs. required python version
    check_version(platform.python_version(), minimum, name='Python ', hard=True)


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'  # string
    if hard:
        assert result, s  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(s)
    return result


@try_except
def check_requirements(requirements=ROOT / 'requirements.txt', exclude=(), install=True, cmds=()):
    """用在train.py、val.py、detect.py等文件
    用于检查已经安装的包是否满足requirements对应txt文件的要求
    Check installed dependencies meet requirements (pass *.txt file or list of packages)
    """
    # Check installed dependencies meet requirements (pass *.txt file or list of packages)
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()  # check python version
    # 解析requirements.txt中的所有包 解析成list 里面存放着一个个的pkg_resources.Requirement类
    # 如: ['matplotlib>=3.2.2', 'numpy>=1.18.5', ……]
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        # 将str字符串requirements转换成路径requirements
        file = Path(requirements)
        assert file.exists(), f"{prefix} {file.resolve()} not found, check failed."
        with file.open() as f:
            requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # number of packages updates
    for i, r in enumerate(requirements):
        try:
            pkg.require(r)
        except Exception:  # DistributionNotFound or VersionConflict if requirements not met
            s = f"{prefix} {r} not found and is required by YOLOv5"
            if install and AUTOINSTALL:  # check environment variable
                LOGGER.info(f"{s}, attempting auto-update...")
                try:
                    assert check_online(), f"'pip install {r}' skipped (offline)"
                    LOGGER.info(check_output(f"pip install '{r}' {cmds[i] if cmds else ''}", shell=True).decode())
                    n += 1
                except Exception as e:
                    LOGGER.warning(f'{prefix} {e}')
            else:
                LOGGER.info(f'{s}. Please install and rerun your command.')

    if n:  # if packages updated
        source = file.resolve() if 'file' in locals() else requirements
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        LOGGER.info(emojis(s))


def check_img_size(imgsz, s=32, floor=0):
    """这个函数主要用于train.py中和detect.py中  用来检查图片的长宽是否符合规定
    检查img_size是否能被s整除，这里默认s为32  返回大于等于img_size且是s的最小倍数
    Verify img_size is a multiple of stride s
    """
    # 取大于等于x的最小值且该值能被divisor整除
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size


def check_imshow():
    """用在detect.py中  使用webcam的时候调用
    检查当前环境是否可以使用opencv.imshow显示图片
    主要有两点限制: Docker环境 + Google Colab环境
    """
    # Check if environment supports image displays
    try:
        # 检查当前环境是否是一个Docker环境 cv2.imshow()不能再docker环境中使用
        assert not is_docker(), 'cv2.imshow() is disabled in Docker environments'
        # 检查当前环境是否是一个Google Colab环境 cv2.imshow()不能在Google Colab环境中使用
        assert not is_colab(), 'cv2.imshow() is disabled in Google Colab environments'
        # 初始化一张图片检查下opencv是否可用
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        LOGGER.warning(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False


def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffix 后缀
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def check_yaml(file, suffix=('.yaml', '.yml')):
    # Search/download YAML file (if necessary) and return path, checking suffix
    return check_file(file, suffix)


def check_file(file, suffix=''):
    """用在train.py、detect.py、test.py等文件中检查本地有没有这个文件
    检查相关文件路径能否找到文件 并返回文件名
    Search/download file (if necessary) and return path
    """
    # Search/download file (if necessary) and return path
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if Path(file).is_file() or file == '':  # exists
        return file
    # 如果传进来的以 'http:/' 或者 'https:/' 开头的url地址, 就下载
    elif file.startswith(('http:/', 'https:/')):  # download
        url = str(Path(file)).replace(':/', '://')  # Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file).split('?')[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if Path(file).is_file():
            LOGGER.info(f'Found {url} locally at {file}')  # file already exists
        else:
            LOGGER.info(f'Downloading {url} to {file}...')
            # 使用torch.hub.download_url_to_file从url地址上中下载文件名为file的文件
            torch.hub.download_url_to_file(url, file)
            # 检查是否下载成功
            assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
        return file
    else:  # search
        # 否则, 传进来的就是当前项目下的一个全局路径 查找匹配的文件名 返回第一个
        # glob.glob: 匹配当前项目下的所有项目 返回所有符合条件的文件files
        files = []
        for d in 'data', 'models', 'utils':  # search directories
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  # find file
        assert len(files), f'File not found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def check_font(font=FONT, progress=False):
    # Download font to CONFIG_DIR if necessary
    font = Path(font)
    file = CONFIG_DIR / font.name
    if not font.exists() and not file.exists():
        url = "https://ultralytics.com/assets/" + font.name
        LOGGER.info(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=progress)


def check_dataset(data, autodownload=True):
    # Download and/or unzip dataset if not found locally
    # Usage: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip

    # Download (optional)
    """用在train.py和detect.py中 检查本地有没有数据集
    检查数据集 如果本地没有则从torch库中下载并解压数据集
    :params data: 是一个解析过的data_dict   len=7
                  例如: ['path'='../datasets/coco128', 'train','val', 'test', 'nc', 'names', 'download']
    :params autodownload: 如果本地没有数据集是否需要直接从torch库中下载数据集  默认True
    """
    extract_dir = ''
    if isinstance(data, (str, Path)) and str(data).endswith('.zip'):  # i.e. gs://bucket/dir/coco128.zip
        download(data, dir=DATASETS_DIR, unzip=True, delete=False, curl=False, threads=1)
        data = next((DATASETS_DIR / Path(data).stem).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        with open(data, errors='ignore') as f:
            data = yaml.safe_load(f)  # dictionary

    # Resolve paths
    path = Path(extract_dir or data.get('path') or '')  # optional 'path' default to '.'
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]

    # Parse yaml
    assert 'nc' in data, "Dataset 'nc' key missing."
    if 'names' not in data:
        data['names'] = [f'class{i}' for i in range(data['nc'])]  # assign class names if missing
    # train: 训练路径  '..\\datasets\\coco128\\images\\train2017'
    # val: 验证路径    '..\\datasets\\coco128\\images\\train2017'
    # test: 测试路径   None
    # s: 下载地址      'https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip'
    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
    if val:
        # path.resolve() 该方法将一些的 路径/路径段 解析为绝对路径
        # val: [WindowsPath('E:/yolo_v5/datasets/coco128/images/train2017')]
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        # 如果val不存在 说明本地不存在数据集
        if not all(x.exists() for x in val):
            LOGGER.info(emojis('\nDataset not found ⚠, missing paths %s' % [str(x) for x in val if not x.exists()]))
            # 如果下载地址s和下载标记(flag)autodownload不为空, 就直接下载
            if s and autodownload:  # download script
                t = time.time()
                root = path.parent if 'path' in data else '..'  # unzip directory i.e. '../'
                # 如果下载地址s是http开头就从url中下载数据集
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    # f: 得到下载文件的文件名 filename
                    f = Path(s).name  # filename
                    LOGGER.info(f'Downloading {s} to {f}...')
                    # 开始下载 利用torch.hub.download_url_to_file函数从s路径中下载文件名为f的文件
                    torch.hub.download_url_to_file(s, f)
                    Path(root).mkdir(parents=True, exist_ok=True)  # create root
                    ZipFile(f).extractall(path=root)  # unzip
                    Path(f).unlink()  # remove zip
                    # 执行解压命名 将文件f解压到root地址 解压后文件名为f
                    r = None  # success
                # 如果下载地址s是bash开头就使用bash指令下载数据集
                elif s.startswith('bash '):  # bash script
                    LOGGER.info(f'Running {s} ...')
                    # 使用bash命令下载
                    r = os.system(s)
                # 否则下载地址就是一个python脚本 执行python脚本下载数据集
                else:  # python script
                    r = exec(s, {'yaml': data})  # return None
                dt = f'({round(time.time() - t, 1)}s)'
                s = f"success ✅ {dt}, saved to {colorstr('bold', root)}" if r in (0, None) else f"failure {dt} ❌"
                LOGGER.info(emojis(f"Dataset download {s}"))
            else:
                # 下载地址为空 或者不需要下载 标记(flag)autodownload
                raise Exception(emojis('Dataset not found ❌'))

    check_font('Arial.ttf' if is_ascii(data['names']) else 'Arial.Unicode.ttf', progress=True)  # download fonts
    return data  # dictionary


def url2file(url):
    # Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt
    url = str(Path(url)).replace(':/', '://')  # Pathlib turns :// -> :/
    file = Path(urllib.parse.unquote(url)).name.split('?')[0]  # '%2F' to '/', split https://url.com/file.txt?auth
    return file


def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1, retry=3):
    # Multi-threaded file download and unzip function, used in data.yaml for autodownload
    """在voc.yaml中下载数据集
    Multi-threaded file download and unzip function
    :params url: 下载文件的url地址
    :params dir: 下载下来文件保存的目录
    :params unzip: 下载后文件是否需要解压
    :params delete: 解压后原文件(未解压)是否需要删除
    :params curl: 是否使用cmd curl语句下载文件  False就使用torch.hub下载
    :params threads: 下载一个文件需要的线程数
    """
    def download_one(url, dir):
        # Download 1 file
        """
        Download 1 file
        :params url: 文件下载地址  Path(url).name=文件名
        :params dir: 文件保存的目录
        """
        success = True
        f = dir / Path(url).name  # filename
        if Path(url).is_file():  # exists in current path
            Path(url).rename(f)  # move to dir
        elif not f.exists():
            LOGGER.info(f'Downloading {url} to {f}...')
            for i in range(retry + 1):
                if curl:
                    s = 'sS' if threads > 1 else ''  # silent
                    r = os.system(f"curl -{s}L '{url}' -o '{f}' --retry 9 -C -")  # curl download
                    success = r == 0
                else:
                    torch.hub.download_url_to_file(url, f, progress=threads == 1)  # torch download
                    success = f.is_file()
                if success:
                    break
                elif i < retry:
                    LOGGER.warning(f'Download failure, retrying {i + 1}/{retry} {url}...')
                else:
                    LOGGER.warning(f'Failed to download {url}...')

        if unzip and success and f.suffix in ('.zip', '.gz'):
            LOGGER.info(f'Unzipping {f}...')
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)  # unzip
            elif f.suffix == '.gz':
                os.system(f'tar xfz {f} --directory {f.parent}')  # unzip
            if delete:
                f.unlink()  # remove zip

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:  # 使用线程池
        # 定义了一个线程池, 最多创建threads个线程
        pool = ThreadPool(threads)
        # 进程池中的该方法会将 iterable 参数传入的可迭代对象分成 chunksize 份传递给不同的进程来处理。
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multi-threaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    """在datasets.py中的LoadStreams类中被调用
    字符串s里在pattern中字符替换为下划线_  注意pattern中[]不能省
    Cleans a string by replacing special characters with underscore _
    """
    # re: 用来匹配字符串（动态、模糊）的模块  正则表达式模块
    # pattern: 表示正则中的模式字符串  repl: 就是replacement的字符串  string: 要被处理, 要被替换的那个string字符串
    # 所以这句话执行的是将字符串s里在pattern中的字符串替换为 "_"
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    """用在train.py中的学习率衰减策略模块
    one_cycle lr  lr先增加, 再减少, 再以更小的斜率减少
    论文: https://arxiv.org/pdf/1803.09820.pdf
    """
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    """用在train.py中  得到每个类别的权重   标签频率高的类权重低
    从训练(gt)标签获得每个类的权重  标签频率高的类权重低
    Get class weights (inverse frequency) from training labels
    :params labels: gt框的所有真实标签labels
    :params nc: 数据集的类别数
    :return torch.from_numpy(weights): 每一个类别根据labels得到的占比(次数越多权重越小) tensor
    """
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    # classes: 所有标签对应的类别labels   labels[:, 0]: 类别   .astype(np.int): 取整
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    # weight: 返回每个类别出现的次数 [1, nc]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    # 将出现次数为0的类别权重全部取1  replace empty bins with 1
    weights[weights == 0] = 1  # replace empty bins with 1
    # 其他所有的类别的权重全部取次数的倒数  number of targets per class
    weights = 1 / weights  # number of targets per class
    # normalize 求出每一类别的占比
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)   # numpy -> tensor


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class_weights and image contents
    """用在train.py中 利用上面得到的每个类别的权重得到每一张图片的权重  再对图片进行按权重进行采样
    通过每张图片真实gt框的真实标签labels和上一步labels_to_class_weights得到的每个类别的权重进行采样
    Produces image weights based on class_weights and image contents
    :params labels: 每张图片真实gt框的真实标签
    :params nc: 数据集的类别数 默认80
    :params class_weights: [80] 上一步labels_to_class_weights得到的每个类别的权重
    """
    # class_counts: 每个类别出现的次数  [num_labels, nc]  每一行是当前这张图片每个类别出现的次数  num_labels=图片数量=label数量
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int), minlength=nc) for x in labels])
    # [80] -> [1, 80]
    # 整个数据集的每个类别权重[1, 80] *  每张图片的每个类别出现的次数[num_labels, 80] = 得到每一张图片每个类对应的权重[128, 80]
    # 另外注意: 这里不是矩阵相乘, 是元素相乘 [1, 80] 和每一行图片的每个类别出现的次数 [1, 80] 分别按元素相乘
    # 再sum(1): 按行相加  得到最终image_weights: 得到每一张图片对应的采样权重[128]
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    """用在test.py中   从80类映射到91类的coco索引 取得对应的class id
    将80个类的coco索引换成91类的coco索引
    :return x: 为80类的每一类在91类中的位置
    """
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    """"用在detect.py和test.py中   操作最后, 将预测信息从xyxy格式转为xywh格式 再保存
    Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
    :params x: [n, x1y1x2y2] (x1, y1): 左上角   (x2, y2): 右下角
    :return y: [n, xywh] (x, y): 中心点  wh: 宽高
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    """用在test.py中 操作之前 转为xyxy才可以进行操作
    注意: x的正方向为右面   y的正方向为下面
    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where x1y1=top-left, x2y2=bottom-right
    :params x: [n, xywh] (x, y):
    :return y: [n, x1y1x2y2] (x1, y1): 左上角  (x2, y2): 右下角
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    """用在datasets.py的 LoadImagesAndLabels类的__getitem__函数、load_mosaic、load_mosaic9等函数中
    将xywh(normalized) -> x1y1x2y2   (x, y): 中间点  wh: 宽高   (x1, y1): 左上点  (x2, y2): 右下点
    Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    """用在datasets.py的 LoadImagesAndLabels类的__getitem__函数中
    将 x1y1x2y2 -> xywh(normalized)  (x1, y1): 左上点  (x2, y2): 右下点  (x, y): 中间点  wh: 宽高
    Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    """
    if clip:
        # 是否需要将x的坐标(x1y1x2y2)限定在尺寸(h, w)内
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    """用在datasets.py的load_mosaic和load_mosaic9函数
    xy(normalized) -> xy
    Convert normalized segments into pixel segments, shape (n,2)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    """用在datasets.py文件中的random_perspective函数中
    将一个多边形标签(不是矩形标签  到底是几边形未知)转化为一个矩形标签
    方法: 对多边形所有的点x1y1 x2y2...  获取其中的(x_min,y_min)和(x_max,y_max) 作为矩形label的左上角和右下角
    Convert 1 segment label to 1 box label, applying inside-image constraint
    :params segment: 一个多边形标签 [n, 2] 传入这个多边形n个顶点的坐标
    :params width: 这个多边形所在图片的宽度
    :params height: 这个多边形所在图片的高度
    :return 矩形标签 [1, x_min+y_min+x_max+y_max]
    """
    # 分别获取当前多边形中所有多边形点的x和y坐标
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    """用在datasets.py文件中的verify_image_label函数中
    将多个多边形标签(不是矩形标签  到底是几边形未知)转化为多个矩形标签
    Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    :params segments: [N, cls+x1y1+x2y2 ...]
    :return [N, cls+xywh]
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    """用在datasets.py文件中的random_perspective函数中
    对segment重新采样，比如说segment坐标只有100个，通过interp函数将其采样为n个(默认1000)
    :params segments: [N, x1x2...]
    :params n: 采样个数
    :return segments: [N, n/2, 2]
    """
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    """用在detect.py和test.py中  将预测坐标从feature map映射回原图
    将坐标coords(x1y1x2y2)从img1_shape缩放到img0_shape尺寸
    Rescale coords (xyxy) from img1_shape to img0_shape
    :params img1_shape: coords相对于的shape大小
    :params coords: 要进行缩放的box坐标信息 x1y1x2y2  左上角 + 右下角
    :params img0_shape: 要将coords缩放到相对的目标shape大小
    :params ratio_pad: 缩放比例gain和pad值   None就先计算gain和pad值再pad+scale  不为空就直接pad+scale
    """
    # ratio_pad为空就先算放缩比例gain和pad值 calculate from img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        # gain  = old / new  取高宽缩放比例中较小的,之后还可以再pad  如果直接取大的, 裁剪就可能减去目标
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        # wh padding  wh中有一个为0  主要是pad另一个
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]  # 指定比例
        pad = ratio_pad[1]      # 指定pad值

    # 因为pad = img1_shape - img0_shape 所以要把尺寸从img1 -> img0 就同样也需要减去pad
    # 如果img1_shape>img0_shape  pad>0   coords从大尺寸缩放到小尺寸 减去pad 符合
    # 如果img1_shape<img0_shape  pad<0   coords从小尺寸缩放到大尺寸 减去pad 符合
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    # 缩放scale
    coords[:, :4] /= gain
    # 防止放缩后的坐标过界 边界处直接剪切
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    """用在上面的xyxy2xywhn、save_one_boxd等函数中
    将boxes的坐标(x1y1x2y2 左上角右下角)限定在图像的尺寸(img_shape hw)内
    Clip bounding x1y1x2y2 bounding boxes to image shape (height, width)
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        # .clamp_(min, max): 将取整限定在(min, max)之间, 超出这个范围自动划到边界上
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Params:
            prediction: [batch, num_anchors(3个yolo预测层), (x+y+w+h+1+num_classes)] = [1, 18900, 25]  3个anchor的预测结果总和
            conf_thres: 先进行一轮筛选，将分数过低的预测框（<conf_thres）删除（分数置0）
            iou_thres: iou阈值, 如果其余预测框与target的iou>iou_thres, 就将那个预测框置0
            classes: 是否nms后只保留特定的类别 默认为None
            agnostic: 进行nms是否也去除不同类别之间的框 默认False
            multi_label: 是否是多标签  nc>1  一般是True
            labels: {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
            max_det: 每张图片的最大目标个数 默认1000
            merge: use merge-NMS 多个bounding box给它们一个权重进行融合  默认False

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    # Checks  检查传入的conf_thres和iou_thres两个阈值是否符合范围
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    # Settings   设置一些变量
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.1 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()  # 记录当前时刻时间
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # 第一层过滤 虑除超小anchor标和超大anchor   x=[18900, 25]
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        # 第二层过滤 根据conf_thres虑除背景目标(obj_conf<conf_thres 0.1的目标 置信度极低的目标)  x=[59, 25]
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        # {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
        # Cat apriori labels if autolabelling 自动标注label时调用  一般不用
        # 自动标记在非常高的置信阈值（即 0.90 置信度）下效果最佳,而 mAP 计算依赖于非常低的置信阈值（即 0.001）来正确评估 PR 曲线下的区域。
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls  # v[:, target相应位置cls,其他位置为0]=1
            x = torch.cat((x, v), 0)  # x: [1204, 85] v: [17, 85] => x: [1221, 85]

        # If none remain process next image
        # 经过前两层过滤后如果该feature map没有目标框了，就结束这轮直接进行下一张图
        if not x.shape[0]:
            continue

        # Compute conf  计算conf_score
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            # 第三轮过滤:针对每个类别score(obj_conf * cls_conf) > conf_thres    [59, 6] -> [51, 6]
            # 这里一个框是有可能有多个物体的，所以要筛选
            # nonzero: 获得矩阵中的非0(True)数据的下标  a.t(): 将a矩阵拆开
            # i: 下标 [43]   j: 类别index [43] 过滤了两个score太低的
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            # pred = [43, xyxy+score+class] [43, 6]
            # unsqueeze(1): [43] => [43, 1] add batch dimension
            # box[i]: [43,4] xyxy
            # pred[i, j + 5].unsqueeze(1): [43,1] score  对每个i,取第（j+5）个位置的值（第j个class的值cla_conf）
            # j.float().unsqueeze(1): [43,1] class
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)  # 一个类别直接取分数最大类的即可
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class  是否只保留特定的类别  默认None  不执行这里
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # 检测数据是否为有限数 Apply finite constraint  这轮可有可无，一般没什么用 所以这里给他注释了
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes  # 如果经过第三轮过滤该feature map没有目标框了，就结束这轮直接进行下一张图
            continue
        elif n > max_nms:  # excess boxes  # 如果经过第三轮过滤该feature map还要很多框(>max_nms)   就需要排序
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # 做个切片 得到boxes和scores   不同类别的box位置信息加上一个很大的数但又不同的数c
        # 这样作非极大抑制的时候不同类别的框就不会掺和到一块了  这是一个作nms挺巧妙的技巧
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # 返回nms过滤后的bounding box(boxes)的索引（降序排列）
        # i=tensor([18, 19, 32, 25, 27])   nms后只剩下5个预测框了
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            # bounding box合并  其实就是把权重和框相乘再除以权重之和
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]  # 最终输出   [5, 6]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    """用在train.py模型训练完后
    将optimizer、training_results、updates...从保存的模型文件f中删除
    Strip optimizer from 'f' to finalize training, optionally save as 's'
    :params f: 传入的原始保存的模型文件
    :params s: 删除optimizer等变量后的模型保存的地址 dir
    """
    # x: 为加载训练的模型
    x = torch.load(f, map_location=torch.device('cpu'))
    # 如果模型是ema replace model with ema
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    # 以下模型训练涉及到的若干个指定变量置空
    for k in 'optimizer', 'best_fitness', 'wandb_id', 'ema', 'updates':  # keys
        x[k] = None
    x['epoch'] = -1    # 模型epoch恢复初始值-1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    # 保存模型 x -> s/f
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")


def print_mutation(results, hyp, save_dir, bucket, prefix=colorstr('evolve: ')):
    """用在train.py的进化超参结束后
    打印进化后的超参结果和results到evolve.txt和hyp_evolved.yaml中
    Print mutation results to evolve.txt (for use with train.py --evolve)
    :params hyp: 进化后的超参 dict {28对 key:value}
    :params results: tuple(7)   (mp, mr, map50, map50:95, box_loss, obj_loss, cls_loss)
    :params yaml_file: 要保存的进化后的超参文件名  runs\train\evolve\hyp_evolved.yaml
    :params bucket: ''
    """
    evolve_csv = save_dir / 'evolve.csv'
    evolve_yaml = save_dir / 'hyp_evolve.yaml'
    keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss',
            'val/obj_loss', 'val/cls_loss') + tuple(hyp.keys())  # [results + hyps]
    keys = tuple(x.strip() for x in keys)
    vals = results + tuple(hyp.values())
    n = len(keys)

    # Download (optional)
    if bucket:
        url = f'gs://{bucket}/evolve.csv'
        if gsutil_getsize(url) > (evolve_csv.stat().st_size if evolve_csv.exists() else 0):
            os.system(f'gsutil cp {url} {save_dir}')  # download evolve.csv if larger than local

    # Log to evolve.csv
    s = '' if evolve_csv.exists() else (('%20s,' * n % keys).rstrip(',') + '\n')  # add header
    with open(evolve_csv, 'a') as f:
        f.write(s + ('%20.5g,' * n % vals).rstrip(',') + '\n')

    # Save yaml  保存yaml配置文件 为'hyp_evolved.yaml'
    with open(evolve_yaml, 'w') as f:
        data = pd.read_csv(evolve_csv)
        data = data.rename(columns=lambda x: x.strip())  # strip keys
        i = np.argmax(fitness(data.values[:, :4]))  #
        generations = len(data)
        f.write('# YOLOv5 Hyperparameter Evolution Results\n' + f'# Best generation: {i}\n' +
                f'# Last generation: {generations - 1}\n' + '# ' + ', '.join(f'{x.strip():>20s}' for x in keys[:7]) +
                '\n' + '# ' + ', '.join(f'{x:>20.5g}' for x in data.values[i, :7]) + '\n\n')
        yaml.safe_dump(data.loc[i][7:].to_dict(), f, sort_keys=False)

    # Print to screen
    LOGGER.info(prefix + f'{generations} generations finished, current result:\n' + prefix +
                ', '.join(f'{x.strip():>20s}' for x in keys) + '\n' + prefix + ', '.join(f'{x:20.5g}'
                                                                                         for x in vals) + '\n\n')

    if bucket:  # 如果需要存到谷歌云盘, 就上传  默认是不需要的
        os.system(f'gsutil cp {evolve_csv} {evolve_yaml} gs://{bucket}')  # upload


def apply_classifier(x, model, img, im0):
    # Apply a second stage classifier to YOLO outputs
    # Example model = torchvision.models.__dict__['efficientnet_b0'](pretrained=True).to(device).eval()
    """用在detect.py文件的nms后继续对feature map送入model2 进行二次分类   几乎不会用它
    定义了一个二级分类器来处理yolo的输出  当前实现本质上是一个参考起点，您可以使用它自行实现此项
    比如你有照片与汽车与车牌, 你第一次剪切车牌, 并将其发送到第二阶段分类器, 以检测其中的字符
    Apply a second stage classifier to yolo outputs
    https://github.com/ultralytics/yolov5/issues/2700  这个函数使用起来很容易出错 不是很推荐使用
    https://github.com/ultralytics/yolov5/issues/1472
    :params x: yolo层的输出
    :params model: 分类模型
    :params img: 进行resize + pad之后的图片
    :params im0: 原尺寸的图片
    """
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()  # 在之前的yolo模型预测的类别
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('example%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            # 用model模型进行分类预测
            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            # 保留预测一致的结果
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    """这是个用处特别广泛的函数 train.py、detect.py、test.py等都会用到
    递增路径 如 run/train/exp --> runs/train/exp{sep}0, runs/exp{sep}1 etc.
    :params path: window path   run/train/exp
    :params exist_ok: False
    :params sep: exp文件名的后缀  默认''
    :params mkdir: 是否在这里创建dir  False
    """
    path = Path(path)  # os-agnostic
    # 如果该文件夹已经存在 则将路径run/train/exp修改为 runs/train/exp1
    if path.exists() and not exist_ok:
        # path.suffix 得到路径path的后缀  ''
        # .with_suffix 将路径添加一个后缀 ''
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


# OpenCV Chinese-friendly functions ------------------------------------------------------------------------------------
imshow_ = cv2.imshow  # copy to avoid recursion errors


def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)


def imwrite(path, im):
    try:
        cv2.imencode(Path(path).suffix, im)[1].tofile(path)
        return True
    except Exception:
        return False


def imshow(path, im):
    imshow_(path.encode('unicode_escape').decode(), im)


cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow  # redefine

# Variables ------------------------------------------------------------------------------------------------------------
NCOLS = 0 if is_docker() else shutil.get_terminal_size().columns  # terminal window size for tqdm
