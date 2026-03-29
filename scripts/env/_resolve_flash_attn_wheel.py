#!/usr/bin/env python3
"""Resolve a pre-built flash-attn wheel URL from GitHub releases.

Detects Python version, PyTorch version, CUDA version, and CXX11 ABI from
the active environment and finds a matching wheel from the Dao-AILab/flash-attention
GitHub releases.

Exit codes:
    0 — success (wheel URL printed as last line of stdout)
    1 — no matching wheel found (fallback to source build)
    2 — environment detection failed
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

GITHUB_API = "https://api.github.com/repos/Dao-AILab/flash-attention/releases"


def detect_environment() -> dict:
    """Detect Python, PyTorch, CUDA, and ABI info from the active environment."""
    info = {}

    # Python version
    info["python"] = f"cp{sys.version_info.major}{sys.version_info.minor}"

    # PyTorch + CUDA
    try:
        import torch

        info["torch"] = torch.__version__.split("+")[0]
        torch_major_minor = ".".join(info["torch"].split(".")[:2])
        info["torch_short"] = torch_major_minor

        if torch.cuda.is_available():
            # Get CUDA version from torch
            cuda_version = torch.version.cuda
            if cuda_version:
                info["cuda"] = cuda_version.replace(".", "")
            else:
                print("WARNING: torch.version.cuda is None", file=sys.stderr)
                info["cuda"] = None
        else:
            print("WARNING: CUDA not available", file=sys.stderr)
            info["cuda"] = None

        # CXX11 ABI
        info["cxx11_abi"] = str(torch._C._GLIBCXX_USE_CXX11_ABI).lower()
    except ImportError:
        print("ERROR: PyTorch not found in environment", file=sys.stderr)
        sys.exit(2)

    return info


def fetch_releases(tag: str | None = None) -> list[dict]:
    """Fetch release(s) from GitHub API."""
    if tag:
        url = f"{GITHUB_API}/tags/{tag}"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
        with urllib.request.urlopen(req) as resp:
            return [json.loads(resp.read())]
    else:
        # Get latest release
        url = f"{GITHUB_API}/latest"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
        with urllib.request.urlopen(req) as resp:
            return [json.loads(resp.read())]


def find_wheel(releases: list[dict], env: dict) -> str | None:
    """Find a matching wheel URL from release assets."""
    if not env.get("cuda"):
        return None

    # Build patterns to match wheel filenames
    # Example: flash_attn-2.8.3+cu124torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
    python_tag = env["python"]
    cuda_tag = f"cu{env['cuda']}"
    torch_tag = f"torch{env['torch_short']}"
    abi_tag = f"cxx11abi{'TRUE' if env['cxx11_abi'] == 'true' else 'FALSE'}"

    print(f"Looking for: {python_tag}, {cuda_tag}, {torch_tag}, {abi_tag}", file=sys.stderr)

    for release in releases:
        for asset in release.get("assets", []):
            name = asset["name"]
            if not name.endswith(".whl"):
                continue
            if "linux_x86_64" not in name:
                continue

            # Check all components match
            if (
                python_tag in name
                and cuda_tag in name
                and torch_tag in name
                and abi_tag in name
            ):
                return asset["browser_download_url"]

    # Try relaxed CUDA match (e.g., cu124 matches cu12x)
    cuda_major = env["cuda"][:3]  # e.g., "124" -> "12"
    for release in releases:
        for asset in release.get("assets", []):
            name = asset["name"]
            if not name.endswith(".whl"):
                continue
            if "linux_x86_64" not in name:
                continue

            cuda_match = re.search(r"cu(\d+)", name)
            if cuda_match and cuda_match.group(1).startswith(cuda_major):
                if python_tag in name and torch_tag in name and abi_tag in name:
                    print(
                        f"Relaxed CUDA match: {cuda_match.group(0)} for {cuda_tag}",
                        file=sys.stderr,
                    )
                    return asset["browser_download_url"]

    return None


def main():
    parser = argparse.ArgumentParser(description="Resolve flash-attn wheel URL")
    parser.add_argument("--tag", type=str, default=None, help="GitHub release tag (e.g., v2.8.3)")
    parser.add_argument("--version", type=str, default=None, help="Alias for --tag")
    args = parser.parse_args()

    tag = args.tag or args.version

    # Detect environment
    env = detect_environment()
    print(f"Python: {env['python']}", file=sys.stderr)
    print(f"PyTorch: {env['torch']} (short: {env['torch_short']})", file=sys.stderr)
    print(f"CUDA: {env.get('cuda', 'N/A')}", file=sys.stderr)
    print(f"CXX11 ABI: {env['cxx11_abi']}", file=sys.stderr)

    if not env.get("cuda"):
        print("No CUDA detected — cannot resolve pre-built wheel.", file=sys.stderr)
        sys.exit(1)

    # Fetch releases
    try:
        releases = fetch_releases(tag)
        print(f"Release: {releases[0]['tag_name']}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to fetch releases: {e}", file=sys.stderr)
        sys.exit(1)

    # Find matching wheel
    url = find_wheel(releases, env)
    if url:
        print(url)  # stdout — captured by the shell script
    else:
        print("No matching pre-built wheel found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
