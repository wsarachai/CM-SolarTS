#!/usr/bin/env python3
"""
Reorganize a Python project directory by moving source and test files
into 'src' and 'tests' directories while preserving the subdirectory structure.
"""

import argparse
from pathlib import Path
import shutil

def reorganize_project(root_dir, dry_run=False):
    """
    Reorganizes the project structure.

    Args:
        root_dir (Path): The root directory of the project.
        dry_run (bool): If True, preview changes without executing them.
    """
    src_dir = root_dir / 'src'
    tests_dir = root_dir / 'tests'

    if not dry_run:
        src_dir.mkdir(exist_ok=True)
        tests_dir.mkdir(exist_ok=True)

    actions = []
    collisions = []

    for path in root_dir.rglob('*.py'):
        # Skip files already in src or tests directories
        if path.is_relative_to(src_dir) or path.is_relative_to(tests_dir):
            continue

        relative_path = path.relative_to(root_dir)
        
        if path.name.startswith('test_'):
            new_path = tests_dir / relative_path
        else:
            new_path = src_dir / relative_path

        if new_path.exists():
            collisions.append(f"Collision detected: {path} -> {new_path}")
            continue

        actions.append((path, new_path))

    if dry_run:
        print("--- Dry Run ---")
        for old, new in actions:
            print(f"Move: {old} -> {new}")
    else:
        for old, new in actions:
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(old, new)
            print(f"Moved: {old} -> {new}")

    if collisions:
        print("\n--- Potential Collisions ---")
        for collision in collisions:
            print(collision)

    print("\n--- Summary ---")
    print(f"Actions to perform: {len(actions)}")
    print(f"Potential collisions: {len(collisions)}")
    if dry_run:
        print("No files were moved.")
    else:
        print("Reorganization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize Python project directory.")
    parser.add_argument('root_dir', type=Path, help="The root directory of the project.")
    parser.add_argument('--dry-run', action='store_true', help="Preview changes without moving files.")
    args = parser.parse_args()

    if not args.root_dir.is_dir():
        print(f"Error: {args.root_dir} is not a valid directory.")
        exit(1)

    reorganize_project(args.root_dir, args.dry_run)