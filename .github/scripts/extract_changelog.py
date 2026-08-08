#!/usr/bin/env python3
"""Print the CHANGELOG.md section for a single version, for use as release notes.

Usage:
    python .github/scripts/extract_changelog.py 0.6.3
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def extract(changelog: str, version: str) -> str:
    """Return the body of the '## [<version>]' section, without its heading.

    Empty subsections (a '### Added' with nothing under it) are dropped so the
    release notes only show categories that actually have entries.
    """
    heading = re.compile(rf"^## \[{re.escape(version)}\]", re.MULTILINE)
    match = heading.search(changelog)
    if match is None:
        raise SystemExit(f"No '## [{version}]' section found in CHANGELOG.md")

    body_start = changelog.index("\n", match.start()) + 1
    next_release = re.compile(r"^## \[", re.MULTILINE).search(changelog, body_start)
    body = changelog[body_start : next_release.start() if next_release else len(changelog)]

    # Split on '### ' subsection headings and keep only the non-empty ones.
    parts = re.split(r"^(### .*)$", body, flags=re.MULTILINE)
    preamble, rest = parts[0], parts[1:]
    kept = [preamble.strip()]
    for title, content in zip(rest[::2], rest[1::2], strict=True):
        if content.strip():
            kept.append(f"{title}\n\n{content.strip()}")

    return "\n\n".join(part for part in kept if part).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="version to extract, without a leading 'v'")
    parser.add_argument(
        "--changelog",
        type=Path,
        default=REPO_ROOT / "CHANGELOG.md",
        help="path to the changelog (default: CHANGELOG.md at the repo root)",
    )
    args = parser.parse_args()

    notes = extract(args.changelog.read_text(encoding="utf-8"), args.version.lstrip("v"))
    if not notes:
        raise SystemExit(f"Section '## [{args.version}]' is empty")

    print(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
