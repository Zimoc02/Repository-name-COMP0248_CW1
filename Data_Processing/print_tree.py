# 打印并保存某个文件夹的目录结构树（用于检查数据结构是否正确）。

import os

ROOT = r"C:\Users\zimoc\Desktop\COMP0248_CW1\COMP0248_set_dataset\full_data\Ali Shihab_8077769_assignsubmission_file_18006111_Shihab"
MAX_DEPTH = 6  # None 表示不限制
IGNORE_FILES = {".DS_Store", "Thumbs.db"}
IGNORE_EXTS = {".tmp"}

OUT_NAME = "_tree.txt"  # 输出文件名（保存在本py同目录）

def should_ignore(name: str) -> bool:
    if name in IGNORE_FILES:
        return True
    _, ext = os.path.splitext(name)
    return ext.lower() in IGNORE_EXTS

def walk_tree(root: str, max_depth=6) -> list[str]:
    lines = []
    root = os.path.abspath(root)
    base_depth = root.rstrip("\\/").count(os.sep)

    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.rstrip("\\/").count(os.sep) - base_depth
        if max_depth is not None and depth > max_depth:
            dirnames[:] = []
            continue

        indent = "  " * depth
        folder_name = os.path.basename(dirpath) if depth > 0 else dirpath
        lines.append(f"{indent}{folder_name}/")

        dirnames[:] = sorted([d for d in dirnames if not should_ignore(d)])
        for f in sorted([f for f in filenames if not should_ignore(f)]):
            lines.append(f"{indent}  {f}")
    return lines

if __name__ == "__main__":
    if not os.path.isdir(ROOT):
        raise FileNotFoundError(f"ROOT not found: {ROOT}")

    tree_lines = walk_tree(ROOT, MAX_DEPTH)

    # ✅ 输出到“这个py文件所在的目录”
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, OUT_NAME)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tree_lines))

    print(f"Saved to: {out_path}")