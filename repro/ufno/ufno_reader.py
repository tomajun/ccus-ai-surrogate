"""
ufno_reader.py —— 按字节偏移直读 U-FNO 官方 .pt，绕开 torch.load(mmap=True) 的段错误
=======================================================================================

为什么要自己读：官方 .pt 单个就有 11GB（test_a）。
  - torch.load(..., mmap=True) 在本机（Windows + torch 2.15）能拿到正确的 shape，
    但**一旦对该张量做任何读取/运算就 Segmentation fault**（clone / !=0 / np.array 全崩）；
  - 不用 mmap 直接 torch.load 又要 11GB 常驻内存，15.8GB 的机器迟早 OOM。

官方用 ZIP_STORED（不压缩）保存，数据段在文件里是裸 float32，可以分页直读。

**必须按元数据里的 size/stride/storage_offset 还原，不能假设 C 连续。**（2026-10-30 实测踩坑）
  sg_test_a.pt : size=(500,96,200,24,12) stride=(5529600,57600,288,12,1)      → 连续，可直接 reshape
  sg_test_u.pt : size=(500,96,200,24)    stride=(460800,200,1,19200)          → **转置视图**！
                 内存里实际按 (时间24, 深度96, 水平200) 排布。
                 若按 (96,200,24) 直接 reshape，得到的是一颗"错位的羽流"：
                 与模型预测的相关性 0.00、rel L2 2.8；按真实 stride 还原后
                 相关性 0.993、rel L2 0.108 —— 差别就在这里。
  sg_val_u.pt  : 同 stride，但 storage_offset=2,073,600,000（=4500 个样本）
                 → 文件里前 8.29GB 是 train 的输出，最后 921MB 才是 val 的 500 个。
"""
from __future__ import annotations

import io
import pickle
import struct
import zipfile

import numpy as np

DATA_PREFIX = "archive/data/"
_ITEMSIZE = np.dtype(np.float32).itemsize


class _MetaUnpickler(pickle.Unpickler):
    """只从 data.pkl 里取 size / stride / storage_offset / storage key，不重建张量。"""

    def __init__(self, buf):
        super().__init__(buf)
        self.size = None
        self.stride = None
        self.storage_offset = 0
        self.storage_key = None
        self.storage_nbytes = None

    def persistent_load(self, pid):
        # torch zip 格式：('storage', dtype, key, size, location)
        if isinstance(pid, (tuple, list)) and len(pid) >= 3:
            self.storage_key = pid[2]
            if len(pid) >= 4:
                self.storage_nbytes = pid[3]
        return pid

    def find_class(self, module, name):
        if name == "_rebuild_tensor_v2":
            def _rebuild(storage, storage_offset, size, stride, requires_grad, hooks):
                self.size = tuple(int(v) for v in size)
                self.stride = tuple(int(v) for v in stride)
                self.storage_offset = int(storage_offset)
                return storage
            return _rebuild
        return lambda *a, **k: None


def tensor_meta(path: str):
    """读出张量元数据 (size, stride, storage_offset, storage_key)。"""
    with zipfile.ZipFile(path) as z:
        pkl = z.read("archive/data.pkl")
    up = _MetaUnpickler(io.BytesIO(pkl))
    up.load()
    if up.size is None or up.stride is None:
        raise RuntimeError(f"{path}: 没能从 data.pkl 解析出 size/stride")
    return up.size, up.stride, up.storage_offset, up.storage_key


def _data_start(path: str, storage_key):
    """返回数据段在文件里的起始字节偏移。优先按 storage key 定位，兜底取最大条目。"""
    with zipfile.ZipFile(path) as z:
        candidates = [i for i in z.infolist() if i.filename.startswith(DATA_PREFIX)]
        if not candidates:
            raise RuntimeError(f"{path} 里找不到张量数据段")
        best = None
        if storage_key is not None:
            want = f"{DATA_PREFIX}{storage_key}"
            best = next((i for i in candidates if i.filename == want), None)
        if best is None:
            best = max(candidates, key=lambda i: i.file_size)
        if best.compress_type != 0:
            raise RuntimeError("该 .pt 的数据段被压缩过，无法按偏移直读，请改用 torch.load")
        with open(path, "rb") as fh:
            fh.seek(best.header_offset)
            lh = fh.read(30)
            if lh[:4] != b"PK\x03\x04":
                raise RuntimeError("本地文件头校验失败，偏移计算不成立")
            nlen, elen = struct.unpack("<HH", lh[26:30])
            start = best.header_offset + 30 + nlen + elen
        return start, best.file_size // _ITEMSIZE


def open_tensor(path: str):
    """按元数据还原成只读的 numpy 视图（可能是非连续的 strided view）。"""
    size, stride, storage_offset, key = tensor_meta(path)
    start, nel = _data_start(path, key)
    need = storage_offset + sum((s - 1) * s_ for s, s_ in zip(size, stride)) + 1
    if need > nel:
        raise ValueError(f"{path}: 声称需要 {need} 个元素，数据段只有 {nel} 个 —— 文件不完整/被截断")
    # 末尾可能有非 4 字节对齐的填充，按整元素数限定映射范围
    flat = np.memmap(path, dtype=np.float32, mode="r", offset=start, shape=(nel,))
    # numpy 的 as_strided 没有 offset 参数，先按 storage_offset 切片（仍是视图）
    view = np.lib.stride_tricks.as_strided(
        flat[storage_offset:], shape=size, strides=tuple(s * _ITEMSIZE for s in stride),
        writeable=False,  # 底层是 mode="r" 的 mmap，写会直接崩，这里显式禁掉
    )
    # as_strided 返回的视图其 .base 就是上面的 memmap，生命周期由 numpy 自动持有
    return view


def open_split(data_dir: str, split: str):
    """打开 (a, u) 一对，逻辑形状 a=(N,96,200,24,12)、u=(N,96,200,24)；样本数取自元数据。"""
    a = open_tensor(f"{data_dir}/sg_{split}_a.pt")
    u = open_tensor(f"{data_dir}/sg_{split}_u.pt")
    if a.shape[0] != u.shape[0]:
        print(f"[ufno_reader] 注意 sg_{split}_a 有 {a.shape[0]} 个样本，sg_{split}_u 却有 {u.shape[0]} 个 —— "
              f"按两者较小值 {min(a.shape[0], u.shape[0])} 使用")
    n = min(a.shape[0], u.shape[0])
    return a[:n], u[:n]
