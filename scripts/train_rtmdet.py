#!/usr/bin/env python3
"""Train RTMDet on a DaphniaScope dataset export.

    python scripts/train_rtmdet.py --inspect                 # what is in the corpus?
    python scripts/train_rtmdet.py                           # train (MPS if there is one)
    python scripts/train_rtmdet.py --epochs 40 --max-steps 20 # a quick smoke run
    python scripts/train_rtmdet.py --evaluate                # score the checkpoint

The corpus is the folder the Dataset surface exports (``labels.json`` +
``images/`` + ``splits/``) - ``datasets/bb`` by default. Everything else lives in
:mod:`daphnia.vision.rtmdet.train`, so this file is only the command line.

Requires PyTorch (and OpenCV) in the environment that runs it, not in the app
environment. See ``docs/rtmdet.md``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daphnia.vision.rtmdet.train import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
