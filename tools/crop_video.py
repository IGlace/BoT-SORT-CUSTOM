#!/usr/bin/env python3
"""Crop a video between two timestamps and save to specified output.

Usage examples:
    python tools/crop_video.py input.mp4 10 20 output.mp4
    python tools/crop_video.py --start 5 --end 15 -i input.mp4 -o out.mp4 --copy

This script requires ffmpeg to be installed and available on PATH.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Crop a video from start second to end second.")
    p.add_argument("input", nargs="?", help="Path to input video file")
    p.add_argument("start", nargs="?", type=float, help="Start time in seconds (inclusive)")
    p.add_argument("end", nargs="?", type=float, help="End time in seconds (exclusive)")
    p.add_argument("output", nargs="?", help="Path to output video file")

    p.add_argument("-i", "--input-file", help="Path to input video file (alternative)")
    p.add_argument("-s", "--start-time", type=float, help="Start time in seconds")
    p.add_argument("-e", "--end-time", type=float, help="End time in seconds")
    p.add_argument("-o", "--output-file", help="Path to output video file")
    p.add_argument("--copy", action="store_true", help="Copy codec (fast) instead of re-encoding")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if exists")
    p.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable to use (default: ffmpeg)")
    return p.parse_args()


def find_ffmpeg(ffmpeg_cmd: str) -> str:
    path = shutil.which(ffmpeg_cmd)
    if path is None:
        print(f"Error: '{ffmpeg_cmd}' not found in PATH. Please install ffmpeg.", file=sys.stderr)
        sys.exit(2)
    return path


def build_ffmpeg_command(ffmpeg_path: str, input_path: Path, output_path: Path, start: float, end: float, copy: bool, overwrite: bool):
    # We'll use -ss (start) and -to (end) in input position for accurate cutting when re-encoding.
    duration = end - start
    if duration <= 0:
        raise ValueError("End time must be greater than start time")

    cmd = [ffmpeg_path]
    if not overwrite:
        cmd += ["-n"]  # no overwrite
    else:
        cmd += ["-y"]

    # Seeking before -i is fast but less accurate for some formats; modern ffmpeg supports -ss before -i for speed.
    # We'll place -ss before -i for speed and then use -to after -i to specify end time relative to start.
    cmd += ["-ss", str(start), "-i", str(input_path), "-to", str(end), "-avoid_negative_ts", "make_zero"]

    if copy:
        cmd += ["-c", "copy"]
    else:
        # default: re-encode using libx264 for video and aac for audio if available
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac"]

    cmd += [str(output_path)]
    return cmd


def main():
    args = parse_args()

    # Determine values: prefer explicit flags
    input_file = args.input_file or args.input
    start = args.start_time if args.start_time is not None else args.start
    end = args.end_time if args.end_time is not None else args.end
    output_file = args.output_file or args.output

    if not input_file or start is None or end is None or not output_file:
        print("Error: input, start, end and output must be provided.\n", file=sys.stderr)
        print("Positional usage: python tools/crop_video.py input.mp4 10 20 output.mp4", file=sys.stderr)
        sys.exit(2)

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"Error: input file '{input_path}' does not exist", file=sys.stderr)
        sys.exit(2)

    if output_path.exists() and not args.overwrite:
        print(f"Error: output file '{output_path}' already exists. Use --overwrite to replace.", file=sys.stderr)
        sys.exit(2)

    ffmpeg_path = find_ffmpeg(args.ffmpeg)

    try:
        cmd = build_ffmpeg_command(ffmpeg_path, input_path, output_path, float(start), float(end), args.copy, args.overwrite)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    print("Running:", " ".join(cmd))

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
