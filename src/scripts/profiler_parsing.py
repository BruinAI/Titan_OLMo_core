#!/usr/bin/env python3
import re
import argparse
from operator import itemgetter

def parse_args():
    parser = argparse.ArgumentParser(
        description="Highlight the largest GPU memory allocation increases in a log file."
    )
    parser.add_argument("logfile", help="Path to your log file")
    parser.add_argument(
        "-n", "--top", type=int, default=5,
        help="How many of the largest increases to show (default: 5)"
    )
    return parser.parse_args()

def extract_entries(path):
    # Matches lines like: [2025-05-20 06:11:31] ... GPU Allocated: 7.5505GB, ...
    pattern = re.compile(
        r'^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*GPU Allocated: (?P<alloc>[0-9.]+)GB'
    )
    entries = []
    with open(path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                ts = m.group("ts")
                alloc = float(m.group("alloc"))
                entries.append((ts, alloc, line.rstrip()))
    return entries

def find_largest_jumps(entries, top_n):
    diffs = []
    for (prev_ts, prev_alloc, _), (curr_ts, curr_alloc, curr_line) in zip(entries, entries[1:]):
        delta = curr_alloc - prev_alloc
        diffs.append({
            "prev_ts": prev_ts,
            "curr_ts": curr_ts,
            "prev_alloc": prev_alloc,
            "curr_alloc": curr_alloc,
            "delta": delta,
            "line": curr_line
        })
    # sort by descending delta
    return sorted(diffs, key=itemgetter("delta"), reverse=True)[:top_n]

def main():
    args = parse_args()
    entries = extract_entries(args.logfile)
    if len(entries) < 2:
        print("Not enough GPU‐allocation entries found.")
        return

    top_jumps = find_largest_jumps(entries, args.top)
    print(f"Top {len(top_jumps)} GPU allocation increases:\n")
    for d in top_jumps:
        print(
            f"{d['prev_ts']} → {d['curr_ts']}: "
            f"{d['prev_alloc']}GB → {d['curr_alloc']}GB   "
            f"(+{d['delta']:.4f}GB)\n  {d['line']}\n"
        )

if __name__ == "__main__":
    main()
