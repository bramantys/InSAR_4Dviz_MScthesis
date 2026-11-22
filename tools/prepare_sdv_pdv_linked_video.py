#!/usr/bin/env python3
"""Prepare linked-video variants compatible with SDV and PDV v1.0.0.

Dependencies: Python 3.8+ (standard library only), FFmpeg, FFprobe.

Example:
    python prepare_sdv_pdv_linked_video.py introduction.mp4 \
        --repo USERNAME/REPOSITORY --tag linked-video-v1

Upload all four files from the output folder as assets of the SAME public
GitHub Release, with the specified tag. Then paste the manifest's asset URL
into SDV or PDV Studio Mode > Linked External Video > Aspect manifest.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

TARGETS = {
    "16:9": (1920, 1080, "linked_intro_16x9.mp4"),
    "1:1": (1080, 1080, "linked_intro_1x1.mp4"),
    "9:16": (1080, 1920, "linked_intro_9x16.mp4"),
}
MANIFEST_NAME = "linked_intro_manifest.json"
# Keep the Proto1/SDV schema: SDV v1.0.0 requires it, and PDV v1.0.0
# explicitly accepts it as a legacy-compatible linked-video manifest.
SCHEMA = "proto1-studio-linked-video-manifest"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Convert one video into 16:9, 1:1 and 9:16 MP4s and an SDV/PDV-compatible manifest."
    )
    parser.add_argument("source", type=Path, help="Source video (e.g., my_intro.mp4)")
    parser.add_argument("--repo", required=True, metavar="OWNER/REPOSITORY",
                        help="Public GitHub repository to host the files")
    parser.add_argument("--tag", required=True, help="GitHub release tag (e.g. linked-video-v1)")
    parser.add_argument("--output-dir", type=Path, default=Path("linked_intro_assets"),
                        help="Directory where the four release assets are written")
    parser.add_argument("--fit", choices=("contain", "cover"), default="contain",
                        help="contain: preserve entire frame with padding (default); "
                             "cover: fill frame by cropping edges")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_file():
        parser.error(f"Source video not found: {source}")

    repo_parts = args.repo.split("/")
    if len(repo_parts) != 2 or not all(repo_parts):
        parser.error("--repo must be OWNER/REPOSITORY (e.g. example/my-videos)")
    import re
    if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", item) for item in repo_parts):
        parser.error("--repo contains unsupported characters")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
        parser.error("Use a simple release tag containing letters, numbers, '.', '_' or '-'")

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        parser.error("FFmpeg and ffprobe must be installed and available on PATH. "
                     "See User Manual Section 2.3, then reopen the terminal.")

    probe = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height", "-of", "json", str(source)],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        parser.error(f"Could not inspect source video: {probe.stderr.strip()}")
    try:
        streams = json.loads(probe.stdout).get("streams", [])
    except json.JSONDecodeError:
        streams = []
    if not streams:
        parser.error("Source file does not contain a video stream")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_url = (f"https://github.com/{repo_parts[0]}/{repo_parts[1]}/releases/download/"
                f"{quote(args.tag, safe='')}/")

    variants = {}
    for ratio, (width, height, filename) in TARGETS.items():
        destination = output / filename
        if destination.resolve() == source:
            parser.error("Source video must not be one of the destination files")

        if args.fit == "contain":
            transform = (f"scale={width}:{height}:force_original_aspect_ratio=decrease:"
                         f"flags=lanczos,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black")
        else:
            transform = (f"scale={width}:{height}:force_original_aspect_ratio=increase:"
                         f"flags=lanczos,crop={width}:{height}")
        video_filter = f"{transform},setsar=1,fps=30,format=yuv420p"
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-stats", "-y",
            "-i", str(source), "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn",
            "-vf", video_filter,
            "-c:v", "libx264", "-preset", "medium", "-crf", "22",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart", "-map_metadata", "-1", str(destination),
        ]
        print(f"\nCreating {ratio} ({width}x{height}): {filename}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print("FFmpeg failed. Check that your FFmpeg build supports libx264.", file=sys.stderr)
            return result.returncode

        probe_out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "json", str(destination)],
            capture_output=True, text=True, check=False,
        )
        if probe_out.returncode != 0:
            print(f"Could not verify {destination}", file=sys.stderr)
            return 1
        stream = json.loads(probe_out.stdout)["streams"][0]
        if (stream.get("width"), stream.get("height")) != (width, height):
            print(f"Incorrect output dimensions for {destination}", file=sys.stderr)
            return 1

        checksum = sha256_file(destination)
        variants[ratio] = {
            "url": base_url + quote(filename),
            "sha256": checksum,
        }
        print(f"  verified · SHA-256: {checksum}", flush=True)

    manifest = {
        "schema": SCHEMA,
        "version": 1,
        "title": source.stem,
        "variants": variants,
    }
    manifest_path = output / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("\nREADY — Upload these four files to the SAME public GitHub Release:")
    for _, _, filename in TARGETS.values():
        print(f"  {filename}")
    print(f"  {MANIFEST_NAME}")
    print("\nManifest URL for SDV and PDV (works after the GitHub Release is published):")
    print(base_url + MANIFEST_NAME)
    print("\nIMPORTANT: Do not edit/re-encode the MP4s after generating the manifest;")
    print("their SHA-256 checksums must continue to match exactly.")
    if args.fit == "contain":
        print("The default contain mode preserves the full frame but may add black bars.")
    else:
        print("Cover mode fills the frame but crops image edges; inspect for lost content.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
