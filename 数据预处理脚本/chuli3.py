#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple, Optional
import shutil
import random

# 可扩展的图片后缀集合
IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def _valid_image_stems(img_dir: Path, exts: Tuple[str, ...]) -> List[str]:
    """收集目录下所有图片文件的 stem（不含后缀），去重后排序。"""
    stems = {p.stem for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts}
    return sorted(stems)


def _choose_split(stems: List[str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """按比例随机划分为 train/val 两个列表（可复现）。"""
    rng = random.Random(seed)
    order = stems[:]  # 拷贝
    rng.shuffle(order)
    cut = int(len(order) * train_ratio)
    return order[:cut], order[cut:]


def _locate_image(img_dir: Path, stem: str, exts: Tuple[str, ...]) -> Optional[Path]:
    """给定 stem，按扩展名顺序查找实际存在的图片路径。"""
    for ext in exts:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _copy_batch(
    stems: Iterable[str],
    img_src: Path,
    lab_src: Path,
    img_dst: Path,
    lab_dst: Path,
    exts: Tuple[str, ...],
) -> int:
    """批量复制（仅当图片与标签均存在时）。返回成功复制的样本数。"""
    count = 0
    for s in stems:
        img = _locate_image(img_src, s, exts)
        if img is None:
            print(f"[跳过] 未找到图片：{s}.*")
            continue

        lab = lab_src / f"{s}.txt"
        if not lab.exists():
            print(f"[跳过] 未找到标签：{lab.name}")
            continue

        try:
            shutil.copy2(img, img_dst / img.name)
            shutil.copy2(lab, lab_dst / lab.name)
            count += 1
        except Exception as e:
            print(f"[错误] 复制样本 {s} 失败：{e}")
    return count


def split_yolo_obb(
    src_images: str | Path,
    src_labels: str | Path,
    out_root: str | Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    exts: Tuple[str, ...] = IMAGE_EXTS,
) -> None:
    """
    将 images/labels 按给定比例拆分到 out_root 下的
      images/{train,val} 与 labels/{train,val}
    仅复制同时具备“图片+同名txt标签”的样本。
    """
    img_dir = Path(src_images)
    lab_dir = Path(src_labels)
    out_dir = Path(out_root)

    # 准备输出目录
    img_train = out_dir / "images" / "train"
    img_val   = out_dir / "images" / "val"
    lab_train = out_dir / "labels" / "train"
    lab_val   = out_dir / "labels" / "val"
    for d in (img_train, img_val, lab_train, lab_val):
        d.mkdir(parents=True, exist_ok=True)

    # 收集样本
    stems = _valid_image_stems(img_dir, exts)
    print(f"发现图片文件：{len(stems)}")
    if not stems:
        print("未检测到可用图片，任务终止。")
        return

    # 划分
    train_stems, val_stems = _choose_split(stems, train_ratio, seed)
    print(f"划分结果 -> 训练: {len(train_stems)} | 验证: {len(val_stems)}")

    # 复制
    print("开始复制训练集样本...")
    n_train = _copy_batch(train_stems, img_dir, lab_dir, img_train, lab_train, exts)

    print("开始复制验证集样本...")
    n_val = _copy_batch(val_stems, img_dir, lab_dir, img_val, lab_val, exts)

    # 汇总
    print("\n=== 完成 ===")
    print(f"训练集复制成功：{n_train}")
    print(f"验证集复制成功：{n_val}")
    print(f"输出根目录：{out_dir.resolve()}")
    print("目录概览：")
    print(f"  images/train  -> {sum(1 for _ in img_train.iterdir())}")
    print(f"  images/val    -> {sum(1 for _ in img_val.iterdir())}")
    print(f"  labels/train  -> {sum(1 for _ in lab_train.iterdir())}")
    print(f"  labels/val    -> {sum(1 for _ in lab_val.iterdir())}")


def main() -> None:
    # 示例路径，可按需修改home/aic/sar
    src_images = "home/aic/sar/split/images"
    src_labels = "home/aic/sar/split/labels"
    out_root   = "home/aic/sar/split"

    if not Path(src_images).exists():
        print(f"[错误] 图片目录不存在：{src_images}")
        return
    if not Path(src_labels).exists():
        print(f"[错误] 标签目录不存在：{src_labels}")
        return

    print("YOLO-OBB 数据集拆分（默认 8:2）")
    
    print("-" * 40)
    split_yolo_obb(src_images, src_labels, out_root, train_ratio=0.8, seed=42)


if __name__ == "__main__":
    main()