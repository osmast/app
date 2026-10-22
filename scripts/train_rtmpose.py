#!/usr/bin/env python3
"""Train RTMPose on a DaphniaScope dataset export's keypoints.

    python scripts/train_rtmpose.py --inspect                 # what is in the corpus?
    python scripts/train_rtmpose.py                           # train (MPS if there is one)
    python scripts/train_rtmpose.py --epochs 40 --max-steps 20 # a quick smoke run
    python scripts/train_rtmpose.py --evaluate                # score the checkpoint

The corpus is the folder the Dataset surface exports (``labels.json`` + ``images/`` +
``splits/``) - ``datasets/kp`` by default. Everything else lives in
:mod:`daphnia.vision.rtmpose.train`, so this file is only the command line.

Requires PyTorch (and OpenCV) in the environment that runs it, not in the app
environment. See ``docs/rtmpose.md``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daphnia.vision.rtmpose.train import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
