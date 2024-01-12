import os
import sys
from pathlib import Path
import logging


def colorstr(*inputs):
    *args, string = inputs if len(inputs) > 1 else ('blue', 'bold', inputs[0])  # color arguments, string
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


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
# 控制台输出
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(fmt="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(console_formatter)
if not LOGGER.handlers:
    LOGGER.addHandler(console_handler)
LOGGER.info(colorstr("green", "LOGGER init!"))


def get_root(file, parent=0):
    """
    获取项目根目录
    :param file: __file__
    :param parent: 父目录层数,0表示当前目录,1表示父目录
    :return: 相对路径
    """
    file = Path(file).resolve()
    root = Path(file).parents[parent]
    if str(root) not in sys.path:
        sys.path.append(str(root))  # add ROOT to PATH
    root = Path(os.path.relpath(root, Path.cwd()))  # relative
    return root


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not Path(p).exists():  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def time_format(s):
    """
    格式化时间
    :param s: 秒数
    :return: 时分秒
    """
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h == 0:
        if m == 0:
            return f"{int(s)} sec"
        else:
            return f"{int(m)}:{int(s)} min"
    return f"{int(h)} hour {int(m)} min {int(s)}sec"
