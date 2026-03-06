import re
import json
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"D:\0248_data_check\Test dataset\COMP0248_Test_data_23")

OUT_DIR = Path(r"D:\0248_data_check\test_build\COMP0248_Test_data_23")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".png", ".jpg", ".jpeg"}
NPY_EXTS = {".npy"}


FRAME_RE = re.compile(r"frame_(\d+)\.(png|jpg|jpeg|npy)$", re.IGNORECASE)

def rel(p: Path) -> str:
    return p.as_posix()

def build_index(root: Path):
    """
    扫描 root 下的 rgb/depth_raw/annotation 等文件，
    以 (gesture, clip, frame_id) 为 key 进行配对。
    """
    # key: (gesture, clip, frame_id) -> dict of modality paths
    items = defaultdict(dict)

    # 同时记录结构信息（用来保存成“目录索引”）
    manifest = {
        "root": str(root),
        "groups": {},  # gesture -> clip -> stats
        "num_samples": 0,
        "notes": "Auto-generated from folder scan. Keyed by (gesture/clip/frame)."
    }

    # 递归扫所有文件
    for p in root.rglob("*"):
        if not p.is_file():
            continue

        # 期望结构：Gxx_xxx/clipYY/{rgb,depth,depth_raw,annotation}/frame_001.xxx
        parts = p.relative_to(root).parts
        if len(parts) < 4:
            continue

        gesture = parts[0]                # e.g., G01_call
        clip = parts[1]                   # e.g., clip01
        modality = parts[2]               # rgb / depth_raw / annotation / depth ...
        fname = parts[-1]

        m = FRAME_RE.match(fname)
        if not m:
            # 例如 depth_metadata.json 这种，不参与样本配对
            continue

        frame_id = int(m.group(1))
        suffix = p.suffix.lower()

        # 只收我们关心的模态
        if modality == "rgb" and suffix in IMG_EXTS:
            items[(gesture, clip, frame_id)]["rgb"] = p
        elif modality == "annotation" and suffix in IMG_EXTS:
            items[(gesture, clip, frame_id)]["mask"] = p
        elif modality == "depth_raw" and suffix in NPY_EXTS:
            items[(gesture, clip, frame_id)]["depth_raw"] = p
        elif modality == "depth" and suffix in IMG_EXTS:
            # 有些 dataloader 可能用 depth png，你可以保留这个字段备用
            items[(gesture, clip, frame_id)]["depth_png"] = p

        # 更新 manifest 的结构信息
        manifest["groups"].setdefault(gesture, {})
        manifest["groups"][gesture].setdefault(clip, {"modalities": set(), "frames_seen": set()})
        manifest["groups"][gesture][clip]["modalities"].add(modality)
        manifest["groups"][gesture][clip]["frames_seen"].add(frame_id)

    # 把 set 转成 list，方便 json 保存
    for gesture, clips in manifest["groups"].items():
        for clip, info in clips.items():
            info["modalities"] = sorted(list(info["modalities"]))
            info["num_frames_seen"] = len(info["frames_seen"])
            info["frames_seen"] = sorted(list(info["frames_seen"]))

    return items, manifest

def make_label_map(manifest):
    """
    默认：按 gesture 名称排序生成 label id（0..C-1）
    如果你训练时已有固定 mapping，请在这里替换为你训练的 mapping。
    """
    gestures = sorted(manifest["groups"].keys())
    return {g: i for i, g in enumerate(gestures)}

def export_manifest(manifest, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def export_label_map(label_map, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

def export_csv(items, label_map, path: Path):
    """
    这里的列名你需要对齐你项目里 index_keyframes.csv 的真实列名。
    我先给一套常见字段：rgb_path, depth_path, mask_path, y, split
    - depth_path 默认写 depth_raw(.npy)。如果你 dataloader 用 depth png，就改成 depth_png。
    - mask_path 在 test 时如果没有 GT，也可以先留空，但你现在目录里有 annotation。
    """
    fieldnames = ["rgb_path", "depth_path", "mask_path", "y", "split", "gesture", "clip", "frame_id"]

    rows = []
    missing = {"rgb": 0, "depth_raw": 0, "mask": 0}

    for (gesture, clip, frame_id), d in sorted(items.items()):
        if "rgb" not in d:       missing["rgb"] += 1
        if "depth_raw" not in d: missing["depth_raw"] += 1
        if "mask" not in d:      missing["mask"] += 1

        # 只导出“配齐”的样本（最稳）
        if not all(k in d for k in ("rgb", "depth_raw", "mask")):
            continue

        rows.append({
            "rgb_path": str(d["rgb"]),
            "depth_path": str(d["depth_raw"]),
            "mask_path": str(d["mask"]),
            "y": int(label_map[gesture]),
            "split": "test",
            "gesture": gesture,
            "clip": clip,
            "frame_id": frame_id,
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    return len(rows), missing

def main():
    items, manifest = build_index(ROOT)
    label_map = make_label_map(manifest)

    manifest_path = OUT_DIR / "test_manifest.json"
    label_map_path = OUT_DIR / "label_map.json"
    csv_path = OUT_DIR / "test_index.csv"

    export_manifest(manifest, manifest_path)
    export_label_map(label_map, label_map_path)
    n_rows, missing = export_csv(items, label_map, csv_path)

    print("=== DONE ===")
    print("manifest:", manifest_path)
    print("label_map:", label_map_path)
    print("csv:", csv_path)
    print("exported samples:", n_rows)
    print("missing counts (over all keys):", missing)

if __name__ == "__main__":
    main()