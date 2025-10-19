#!/usr/bin/env python3
"""
A script to find and sanitize problematic Unicode characters in source code files.
"""

import os
import argparse
import fnmatch
import shutil

# Define problematic characters and their replacements
# Semicolon, Greek Question Mark (U+037E)
# Space, Non-breaking space (U+00A0)
# Space, Zero-width space (U+200B)
# Hyphen, Non-breaking hyphen (U+2011)
# Single quote, Left single quotation mark (U+2018)
# Single quote, Right single quotation mark (U+2019)
# Double quote, Left double quotation mark (U+201C)
# Double quote, Right double quotation mark (U+201D)
# Bidirectional override, Right-to-Left Override (U+202E)
HOMOGLYPHS = {
    '\u037e': ';',
    '\u00a0': ' ',
    '\u200b': '',
    '\u2011': '-',
    '\u2018': "'",
    '\u2019': "'",
    '\u201c': '"',
    '\u201d': '"',
    '\u202e': '',
}

def find_files(directory, include_patterns, exclude_patterns):
    """
    Recursively find files matching include patterns and excluding exclude patterns.
    """
    for root, dirs, files in os.walk(directory):
        # Exclude directories
        dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, p) for p in exclude_patterns)]
        
        for filename in files:
            if any(fnmatch.fnmatch(filename, p) for p in include_patterns):
                if not any(fnmatch.fnmatch(filename, p) for p in exclude_patterns):
                    yield os.path.join(root, filename)

def sanitize_file(filepath, fix=False):
    """
    Scan a file for problematic characters and optionally fix them.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Error: Could not decode file {filepath} as UTF-8.")
        return False

    found_problems = False
    new_content = content
    
    for i, line in enumerate(content.splitlines()):
        for j, char in enumerate(line):
            if char in HOMOGLYPHS:
                found_problems = True
                print(f"Found problematic character U+{ord(char):04X} in {filepath} at line {i+1}, col {j+1}")
                if fix:
                    new_content = new_content.replace(char, HOMOGLYPHS[char])

    if fix and found_problems:
        # Create a backup
        shutil.copy(filepath, filepath + '.bak')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Sanitized {filepath}")

    return found_problems

def main():
    parser = argparse.ArgumentParser(description="Sanitize source code files.")
    parser.add_argument('directory', help="The directory to scan.")
    parser.add_argument('--include', nargs='+', default=['*.py', '*.js', '*.c'], help="File patterns to include.")
    parser.add_argument('--exclude', nargs='+', default=['.git', 'node_modules', '*.min.js', 'package-lock.json'], help="Directory/file patterns to exclude.")
    parser.add_argument('--fix', action='store_true', help="Fix files in place (creates .bak backups).")
    
    args = parser.parse_args()

    found_any_problems = False
    for filepath in find_files(args.directory, args.include, args.exclude):
        if sanitize_file(filepath, args.fix):
            found_any_problems = True

    if found_any_problems and not args.fix:
        print("\nFound problematic characters. Run with --fix to correct them.")
        exit(1)
    elif not found_any_problems:
        print("\nNo problematic characters found.")

if __name__ == "__main__":
    main()