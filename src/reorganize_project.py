#!/usr/bin/env python3
"""
Reorganize a Python project directory by creating standardized directories for 
source code, tests, documentation, data, and configuration, then moving files 
into these directories based on predefined file type mappings while preserving 
git history for version-controlled files using git mv and handling name conflicts 
by appending numerical suffixes while ignoring virtual environments and build artifacts.
"""

import argparse
import os
import subprocess
import shutil
from pathlib import Path
import re


def is_git_repo(directory):
    """Check if directory is a git repository."""
    return (directory / '.git').exists()


def git_mv(src, dst):
    """Move file using git mv to preserve history."""
    try:
        subprocess.run(['git', 'mv', str(src), str(dst)], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def resolve_conflict(dst_path):
    """Resolve name conflicts by appending numerical suffixes."""
    if not dst_path.exists():
        return dst_path
    
    base = dst_path.stem
    extension = dst_path.suffix
    parent = dst_path.parent
    
    counter = 1
    while True:
        new_name = f"{base}_{counter}{extension}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_path
        counter += 1


def should_ignore(path, root_dir):
    """Check if path should be ignored (virtual environments, build artifacts, etc.)."""
    path_str = str(path)
    
    # Common patterns to ignore
    ignore_patterns = [
        r'\.git',
        r'\.idea',
        r'\.vscode',
        r'\.pytest_cache',
        r'\.mypy_cache',
        r'__pycache__',
        r'node_modules',
        r'venv',
        r'env',
        r'\.env',
        r'build',
        r'dist',
        r'\.eggs',
        r'egg-info',
        r'\.tox',
        r'\.coverage',
        r'htmlcov',
        r'\.DS_Store',
        r'Thumbs\.db'
    ]
    
    for pattern in ignore_patterns:
        if re.search(pattern, path_str):
            return True
    
    # Check if path is in a standard directory that shouldn't be reorganized
    relative_path = path.relative_to(root_dir)
    if relative_path.parts[0] in ['src', 'tests', 'docs', 'data', 'config', 'latex']:
        return True
    
    return False


def get_file_category(file_path, root_dir):
    """Determine the category directory for a file based on its type and location."""
    relative_path = file_path.relative_to(root_dir)
    
    # File type mappings
    source_extensions = {'.py', '.pyx', '.pxd', '.pyi'}
    test_patterns = ['test_', '_test.py', 'conftest.py']
    doc_extensions = {'.md', '.rst', '.txt', '.tex', '.bib', '.pdf', '.doc', '.docx'}
    data_extensions = {'.csv', '.json', '.xml', '.yml', '.toml', '.ini', '.cfg', 
                       '.nc', '.grib', '.hdf', '.h5', '.mat', '.zip', '.tar', '.gz', '.bz2'}
    config_extensions = {'.ini', '.cfg', '.conf', '.toml', '.json'}
    
    # Check if it's a test file
    if (file_path.name.startswith('test_') or 
        file_path.name.endswith('_test.py') or 
        file_path.name == 'conftest.py'):
        return 'tests'
    
    # Check if it's already in a standard directory
    if relative_path.parts[0] in ['src', 'tests', 'docs', 'data', 'config']:
        return None
    
    # Check file extension
    ext = file_path.suffix.lower()
    
    if ext in source_extensions:
        return 'src'
    elif ext in doc_extensions:
        return 'docs'
    elif ext in data_extensions:
        return 'data'
    elif ext in config_extensions:
        return 'config'
    
    # Special cases
    if file_path.name in ['README', 'LICENSE', 'CHANGELOG', 'CONTRIBUTING']:
        return 'docs'
    if file_path.name in ['requirements.txt', 'setup.py', 'setup.cfg', 'pyproject.toml']:
        return 'config'
    
    # Default to data for unknown file types
    return 'data'


def reorganize_project(root_dir, dry_run=False):
    """
    Reorganizes the project structure.

    Args:
        root_dir (Path): The root directory of the project.
        dry_run (bool): If True, preview changes without executing them.
    """
    # Create standard directories
    standard_dirs = ['src', 'tests', 'docs', 'data', 'config']
    
    if not dry_run:
        for dir_name in standard_dirs:
            (root_dir / dir_name).mkdir(exist_ok=True)
    
    actions = []
    collisions = []
    git_available = is_git_repo(root_dir)
    
    # Find all files to potentially move
    for path in root_dir.rglob('*'):
        if not path.is_file():
            continue
            
        if should_ignore(path, root_dir):
            continue
            
        category = get_file_category(path, root_dir)
        if not category:
            continue
            
        # Calculate new path
        relative_path = path.relative_to(root_dir)
        new_path = root_dir / category / relative_path
        
        # Resolve conflicts
        if new_path.exists():
            new_path = resolve_conflict(new_path)
            collisions.append(f"Collision resolved: {path} -> {new_path}")
        
        actions.append((path, new_path, git_available))
    
    # Execute or preview actions
    if dry_run:
        print("--- Dry Run ---")
        for old, new, use_git in actions:
            git_note = " (using git mv)" if use_git else ""
            print(f"Move: {old} -> {new}{git_note}")
    else:
        for old, new, use_git in actions:
            # Create parent directories if they don't exist
            new.parent.mkdir(parents=True, exist_ok=True)
            
            # Try to use git mv if available and file is tracked
            if use_git and git_available:
                success = git_mv(old, new)
                if success:
                    print(f"Moved (git): {old} -> {new}")
                    continue
            
            # Fall back to regular move
            shutil.move(old, new)
            print(f"Moved: {old} -> {new}")
    
    # Print summary
    if collisions:
        print("\n--- Resolved Collisions ---")
        for collision in collisions:
            print(collision)
    
    print("\n--- Summary ---")
    print(f"Actions performed: {len(actions)}")
    print(f"Resolved collisions: {len(collisions)}")
    print(f"Git repository detected: {git_available}")
    
    if dry_run:
        print("No files were moved.")
    else:
        print("Reorganization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reorganize Python project directory into standard structure."
    )
    parser.add_argument(
        'root_dir', 
        type=Path, 
        help="The root directory of the project.",
        default=".",
        nargs="?"
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help="Preview changes without moving files."
    )
    args = parser.parse_args()
    
    if not args.root_dir.is_dir():
        print(f"Error: {args.root_dir} is not a valid directory.")
        exit(1)
    
    reorganize_project(args.root_dir, args.dry_run)