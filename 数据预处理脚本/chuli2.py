#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple
import cv2


def _iter_lines(fp: Path) -> Iterable[Tuple[int, str]]:
    """按行读取文本，带行号，兼容脏字符。"""
    for i, line in enumerate(fp.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        yield i, line


def _to_normalized_pairs(vals: List[float], w: int, h: int) -> List[float]:
    """
    将 [x1,y1,...,x4,y4] 归一化到 [0,1]，并做裁剪。
    """
    out: List[float] = []
    for i in range(0, 8, 2):
        x = vals[i] / w
        y = vals[i + 1] / h
        # clamp
        x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
        y = 0.0 if y < 0.0 else (1.0 if y > 1.0 else y)
        out.extend([x, y])
    return out


def _process_label_file(label_fp: Path, image_fp: Path) -> List[str]:
    """
    处理单个标签文件，返回归一化后的输出行列表。
    期望行格式：cls x1 y1 x2 y2 x3 y3 x4 y4 （共9个数）
    """
    img = cv2.imread(str(image_fp))
    if img is None:
        raise ValueError(f"无法读取图片：{image_fp}")

    h, w = img.shape[:2]
    outputs: List[str] = []

    for ln, raw in _iter_lines(label_fp):
        s = raw.strip()
        if not s:
            continue

        parts = s.split()
        if len(parts) != 9:
            print(f"警告：{label_fp.name} 第{ln}行应为9个值，已跳过 -> {s}")
            continue

        try:
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
        except ValueError as e:
            print(f"警告：{label_fp.name} 第{ln}行解析失败，已跳过 -> {e}")
            continue

        norm = _to_normalized_pairs(coords, w, h)
        outputs.append(f"{cls} " + " ".join(f"{v:.6f}" for v in norm))

    return outputs


def normalize_yolo_labels(
    images_folder: Path | str,
    labels_folder: Path | str,
    output_labels_folder: Path | str | None = None,
) -> None:
    """
    将 YOLO OBB 标签从像素坐标归一化到 [0,1]，输出到指定目录（默认覆盖原 labels）。
    只匹配同名 .png 图片（与原逻辑一致）。
    """
    img_dir = Path(images_folder)
    lb_dir = Path(labels_folder)
    out_dir = Path(output_labels_folder) if output_labels_folder else lb_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(lb_dir.glob("*.txt"))
    if not label_files:
        print(f"错误：{lb_dir} 下未发现 .txt 标签文件")
        return

    total = len(label_files)
    ok = skipped = failed = 0
    print(f"开始归一化处理，共 {total} 个文件。")

    for idx, lb in enumerate(label_files, 1):
        if idx == 1 or idx % 1000 == 0:
            pct = idx / total * 100
            print(f"进度：{idx}/{total}（{pct:.1f}%）")

        img_fp = img_dir / f"{lb.stem}.png"
        if not img_fp.exists():
            print(f"提示：缺少对应图片 {img_fp.name}，跳过 {lb.name}")
            skipped += 1
            continue

        try:
            lines = _process_label_file(lb, img_fp)
            (out_dir / lb.name).write_text(
                ("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8"
            )
            ok += 1
        except Exception as e:
            failed += 1
            print(f"错误：处理 {lb.name} 失败 -> {e}")

    print("\n=== 完成 ===")
    print(f"成功处理：{ok}")
    print(f"跳过文件：{skipped}（缺图或读图失败）")
    print(f"处理失败：{failed}")
    print(f"输出目录：{out_dir if out_dir != lb_dir else '覆盖原 labels 目录'}")


def normalize_dataset_folders(dataset_folder: Path | str) -> None:
    """
    扫描 dataset_folder/images/{train,val} 与 labels/{train,val}，对存在的子集执行归一化。
    """
    root = Path(dataset_folder)
    for split in ("train", "val"):
        imgs = root / "images" / split
        lbs = root / "labels" / split
        if imgs.exists() and lbs.exists():
            print(f"\n处理子集：{split}")
            normalize_yolo_labels(imgs, lbs)
        else:
            print(f"跳过 {split}：目录不存在")


def main() -> None:
    # 建议使用原始字符串避免反斜杠转义问题
    images_folder = "home\aic\sar\annfiles\images"
    labels_folder = "home\aic\sar\annfiles\labels"
    # 设为 None 表示覆盖原 labels；也可指定其他输出目录
    output_labels_folder = None

    normalize_yolo_labels(images_folder, labels_folder, output_labels_folder)


if __name__ == "__main__":
    main()
