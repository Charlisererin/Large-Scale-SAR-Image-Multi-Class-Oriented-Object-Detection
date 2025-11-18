from pathlib import Path
import pickle
import numpy as np
from ultralytics import YOLO

CLASS_MAP = {
    0: "ship",
    1: "aircraft",
    2: "car",
    3: "tank",
    4: "bridge",
    5: "harbor",
}

def check_model_classes(model_path: str):
    """返回模型类别映射（不打印）。"""
    model = YOLO(model_path)
    return dict(model.names)


def _numeric_key(p: Path):
    try:
        return int(p.stem)
    except ValueError:
        return p.stem


def convert_yolo_to_pkl(
    model_path: str,
    test_dir: str,
    output_pkl_path: str,
    conf_threshold: float = 0.001,
    iou: float = 0.7,
    max_det: int = 600,
    temp_interval: int = 200,  # 每处理多少张保存一次临时文件；<=0 关闭临时保存
):
    """
    运行推理并生成 pkl（静默）。返回 (all_results, summary)。
    summary: {'images': N, 'detections': M, 'errors': K, 'temp_saves': T}
    """
    model = YOLO(model_path)
    test_dir = Path(test_dir)
    out_path = Path(output_pkl_path)
    temp_path = out_path.with_suffix(out_path.suffix + ".temp")

    exts = (".png", ".jpg", ".jpeg")
    imgs = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in exts], key=_numeric_key)

    all_results = []
    temp_saves = 0

    if not imgs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump([], f)
        return [], {"images": 0, "detections": 0, "errors": 0, "temp_saves": 0}

    for i, p in enumerate(imgs, 1):
        entry = {"image": p.name, "poly": np.empty((0, 8), dtype=float), "scores": [], "labels": []}
        try:
            res = model(str(p), conf=conf_threshold, iou=iou, max_det=max_det, verbose=False)[0]
            if hasattr(res, "obb") and res.obb is not None and len(res.obb) > 0:
                polys = res.obb.xyxyxyxy.cpu().numpy()
                confs = res.obb.conf.cpu().numpy()
                clses = res.obb.cls.cpu().numpy()

                keep_poly, keep_scores, keep_labels = [], [], []
                for j in range(len(res.obb)):
                    cls_idx = int(clses[j])
                    if cls_idx in CLASS_MAP:
                        keep_poly.append(polys[j].flatten())
                        keep_scores.append(float(confs[j]))
                        keep_labels.append(CLASS_MAP[cls_idx])

                if keep_poly:
                    entry["poly"] = np.asarray(keep_poly, dtype=float)
                    entry["scores"] = keep_scores
                    entry["labels"] = keep_labels
        except Exception as e:
            entry["error"] = str(e)

        all_results.append(entry)

        if temp_interval and temp_interval > 0 and (i % temp_interval == 0):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, "wb") as f:
                pickle.dump(all_results, f)
            temp_saves += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)

    summary = {
        "images": len(imgs),
        "detections": sum(len(x["scores"]) for x in all_results),
        "errors": sum(1 for x in all_results if "error" in x),
        "temp_saves": temp_saves,
    }
    return all_results, summary


def verify_pkl_format(pkl_path: str):
    """
    验证 pkl 结构（静默），返回统计信息字典；不打印。
    """
    p = Path(pkl_path)
    data = pickle.load(open(p, "rb"))
    stats = {
        "images": len(data),
        "detections": sum(len(d.get("scores", [])) for d in data),
        "has_error_entries": any("error" in d for d in data),
        "sample": [],
        "size_bytes": p.stat().st_size if p.exists() else 0,
    }
    for d in data[:5]:
        stats["sample"].append(
            {
                "image": d.get("image", ""),
                "num": len(d.get("scores", [])),
                "scores_head": d.get("scores", [])[:3],
                "labels_head": d.get("labels", [])[:3],
                "error": d.get("error", None),
            }
        )
    return stats


if __name__ == "__main__":
    model_path = "\ultralytics-main\runs\train\expm\weights\best.pt"
    test_dir = "\test_B_images\images"
    output_pkl = "resalt.pkl"

    # 不打印任何消息；仅在需要时由外部捕获返回值使用
    if Path(model_path).exists() and Path(test_dir).exists():
        _, _ = convert_yolo_to_pkl(
            model_path=model_path,
            test_dir=test_dir,
            output_pkl_path=output_pkl,
            conf_threshold=0.001,
            iou=0.70,
            max_det=600,
            temp_interval=200,  
        )

