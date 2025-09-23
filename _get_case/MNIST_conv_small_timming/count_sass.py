#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count total SASS instruction frequency from cuobjdump/nvdisasm output.

Usage:
    python count_sass_instructions.py /path/to/kernel_sm89.sass
"""

import re
import collections
import sys
import os

# Regular expression to match instructions and remove comments
ADDR_RE = re.compile(r'/\*.*?\*/')  # Remove leading address comment (/*0000*/)

def extract_opcode(line: str) -> str | None:
    """
    Try to extract the opcode token from a SASS line.
    - Removes the leading address comment (/*0000*/)
    - Skips labels at line start ("label:")
    - Skips directives (starting with '.')
    - Handles optional predication like '@P0' / '@!P0'
    Returns opcode (possibly with modifiers) or None if not an instruction line.
    """
    # Quick rejects
    if not line.strip():
        return None

    # Remove the first address blob (/*....*/)
    s = ADDR_RE.sub('', line, count=1).strip()
    if not s:
        return None

    # Skip pure directives/comments
    if s.startswith('.') or s.startswith('//') or s.startswith('/*'):
        return None

    # Only consider lines that contain an instruction terminator ';'
    if ';' not in s:
        return None

    # Tokenize and skip predication tokens like '@P0' or '@!P0'
    tokens = s.split()
    opcode = None
    for tok in tokens:
        if tok.startswith('@'):  # Skip predicates
            continue
        # First non-predicate token is the opcode
        opcode = tok
        break
    if not opcode:
        return None

    # Clean up token: strip trailing punctuation like ',' or ';'
    opcode = opcode.rstrip(',;')

    return opcode


def parse_sass(path: str) -> collections.Counter[str]:
    instruction_counts = collections.Counter()

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Extract opcode from line
            opcode = extract_opcode(line)
            if opcode:
                instruction_counts[opcode] += 1

    return instruction_counts


def print_report(instruction_counts: collections.Counter[str]):
    total_instructions = sum(instruction_counts.values())
    print(f"\n=== Total Instruction Count (Total instructions = {total_instructions}) ===")
    for i, (opcode, count) in enumerate(instruction_counts.most_common(), start=1):
        pct = (count / total_instructions * 100.0) if total_instructions else 0.0
        print(f"{i:>3}. {opcode:<24} {count:>10}  ({pct:5.2f}%)")


def main():
    if len(sys.argv) != 2:
        print("Usage: python count_sass_instructions.py <path-to-sass-file>", file=sys.stderr)
        sys.exit(1)

    sass_file = sys.argv[1]
    if not os.path.isfile(sass_file):
        print(f"Error: The file '{sass_file}' does not exist or is not a valid file.", file=sys.stderr)
        sys.exit(2)

    instruction_counts = parse_sass(sass_file)
    if not instruction_counts:
        print(f"No instructions found. Are you sure this is a cuobjdump/nvdisasm SASS file?", file=sys.stderr)
        sys.exit(3)

    print_report(instruction_counts)


if __name__ == '__main__':
    main()
