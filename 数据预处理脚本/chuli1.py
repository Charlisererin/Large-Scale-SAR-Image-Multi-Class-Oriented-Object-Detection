#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Tuple, List

# 可按需修改：类别到ID
LABEL2ID: Dict[str, int] = {
    "ship": 0,
    "aircraft": 1,
    "car": 2,
    "tank": 3,
    "bridge": 4,
    "harbor": 5,
}
def _parse_line(
    raw: str,
    file_name: str,
    line_no: int,
    label2id: Dict[str, int],
) -> Tuple[bool, str]:
    """
    解析一行 DOTA 标注为 YOLO-OBB 文本。
    返回 (ok, message_or_output)。ok=True 时为输出行；False 为提示信息。
    """
    s = raw.strip()
    if not s:
        return False, ""  # 空行静默跳过

    parts = s.split()
    if len(parts) < 10:
        return False, f"格式有误，已跳过：{file_name} 第{line_no}行 -> {raw}"

    # 坐标（前8个）+ 类别（第9个）
    x1, y1, x2, y2, x3, y3, x4, y4 = parts[:8]
    cls_name = parts[8]

    if cls_name not in label2id:
        return False, f"未知类别，已跳过：{file_name} 第{line_no}行 -> {cls_name}"

    cls_id = label2id[cls_name]
    return True, f"{cls_id} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}"


def _read_lines(fp: Path) -> Iterable[Tuple[int, str]]:
    # 使用 errors='ignore' 兼容脏数据
    text = fp.read_text(encoding="utf-8", errors="ignore")
    for i, line in enumerate(text.splitlines(), start=1):
        yield i, line


def dota2yolo_obb(
    src_dir: Path,
    dst_dir: Path | None = None,
    label2id: Dict[str, int] = LABEL2ID,
) -> None:
    """
    将 src_dir 下的所有 .txt 标注转为 YOLO-OBB，并写入 dst_dir（默认同目录）。
    """
    src = Path(src_dir)
    out_dir = Path(dst_dir) if dst_dir is not None else src
    out_dir.mkdir(parents=True, exist_ok=True)

    files: List[Path] = sorted(src.glob("*.txt"))
    if not files:
        print(f"未在目录中发现 .txt 标注文件：{src.resolve()}")
        return

    n_ok, n_fail = 0, 0
    for fp in files:
        try:
            outputs: List[str] = []
            for ln, raw in _read_lines(fp):
                ok, msg = _parse_line(raw, fp.name, ln, label2id)
                if ok:
                    outputs.append(msg)
                elif msg:  # 非空提示才打印
                    print(msg)

            (out_dir / fp.name).write_text(
                ("\n".join(outputs) + ("\n" if outputs else "")),
                encoding="utf-8",
            )
            n_ok += 1
            print(f"已转换：{fp.name} -> {(out_dir / fp.name).name}（{len(outputs)} 条目标）")
        except Exception as exc:  # 保底防护，单文件失败不影响其他文件
            n_fail += 1
            print(f"处理失败：{fp.name} | {exc}")

    # 汇总
    print("\n=== 处理统计 ===")
    print(f"成功文件数：{n_ok}")
    print(f"失败文件数：{n_fail}")
    print(f"类别映射：{label2id}")


def main() -> None:
    # 注意：避免使用诸如 "\annfiles" 之类的路径写法（会被当作转义）。
    # 推荐使用原始字符串 r"…", 或正斜杠/Path。
    input_folder = "annfiles"   # 示例：r"annfiles"
    output_folder = "labels"     # 示例：r"labels"
    dota2yolo_obb(Path(input_folder), Path(output_folder))


if __name__ == "__main__":
    main()
