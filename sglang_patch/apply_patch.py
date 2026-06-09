#!/usr/bin/env python3
"""
DoLa SGLang Patch 安装/回滚脚本

用法:
    python apply_patch.py --apply     # 应用 DoLa patch
    python apply_patch.py --revert    # 回滚 DoLa patch
    python apply_patch.py --status    # 检查 patch 状态
"""

import os, sys, shutil, argparse
from pathlib import Path

PATCH_FILES = {
    "srt/layers/logits_processor.py": "logits_processor.py",
    "srt/models/qwen3_5.py": "qwen3_5.py",
}

def find_sglang_path():
    try:
        import sglang
        return os.path.dirname(sglang.__file__)
    except ImportError:
        print("错误: 未找到 sglang，请先 pip install sglang")
        sys.exit(1)

def get_full_dir():
    return Path(__file__).parent / "full_files"

def apply_patch():
    sglang_dir = find_sglang_path()
    full_dir = get_full_dir()

    for rel_path, filename in PATCH_FILES.items():
        target = os.path.join(sglang_dir, rel_path)
        backup = target + ".dola_bak"
        source = full_dir / filename

        if not os.path.exists(target):
            print(f"  ⚠️  目标不存在: {target}")
            continue
        if not source.exists():
            print(f"  ⚠️  源不存在: {source}")
            continue

        if not os.path.exists(backup):
            shutil.copy2(target, backup)
            print(f"  📦 备份: {backup}")
        else:
            print(f"  ⏭️  已有备份: {backup}")

        shutil.copy2(source, target)
        print(f"  ✅ 替换: {target}")

    print("\n✅ DoLa patch 已应用，需重启 sglang 服务")

def revert_patch():
    sglang_dir = find_sglang_path()

    for rel_path in PATCH_FILES:
        target = os.path.join(sglang_dir, rel_path)
        backup = target + ".dola_bak"

        if os.path.exists(backup):
            shutil.copy2(backup, target)
            os.remove(backup)
            print(f"  ✅ 恢复: {target}")
        else:
            print(f"  ⏭️  无备份: {target}")

    print("\n✅ DoLa patch 已回滚，需重启 sglang 服务")

def check_status():
    sglang_dir = find_sglang_path()
    for rel_path in PATCH_FILES:
        backup = os.path.join(sglang_dir, rel_path) + ".dola_bak"
        if os.path.exists(backup):
            print(f"  🔧 已 patch: {rel_path}")
        else:
            print(f"  ⬜ 未 patch: {rel_path}")

def main():
    parser = argparse.ArgumentParser(description="DoLa SGLang Patch 管理")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true")
    group.add_argument("--revert", action="store_true")
    group.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.apply:   apply_patch()
    elif args.revert: revert_patch()
    elif args.status: check_status()

if __name__ == "__main__":
    main()
