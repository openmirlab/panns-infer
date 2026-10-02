"""The panns-infer command runs the same session contract and emits full JSON.

The output includes all 527 scores/labels and the complete native embedding.
An optional file target writes the same payload; previews never replace it.
Reads: CLI audio/asset/device options and portable dataclass results.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PANNs CNN14 AudioSet tagging and native2048embedding"
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-config", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--pad-short", action="store_true", help="Right-pad short audio with zeros")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    from . import PANNsSession

    try:
        with PANNsSession(
            device=args.device,
            cache_dir=args.cache_dir,
            checkpoint_dir=args.checkpoint_dir,
            offline=args.offline,
            checkpoint_config=args.checkpoint_config,
        ) as session:
            session.load()
            result = session.infer(args.audio, pad_short=args.pad_short)
        payload = json.dumps(asdict(result), ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"panns-infer: {exc}\n")
    return 0
