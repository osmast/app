#!/usr/bin/env python3
"""Check whether this environment can train: PyTorch, the accelerator, OpenCV.

    python scripts/check_torch.py

Run it in the environment you intend to train in (``z-app-train``), *before* a long
run. It prints the torch build, whether **MPS** (the Apple-silicon GPU) is available,
where a run would put a checkpoint, and then does a tiny forward/backward pass on
MPS so "available" is proven rather than assumed - a Metal build can report
availability and still fail on an operation.

Exit code 0 means training can start; 1 means something must be installed or fixed
first, and the message says what.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    print("DaphniaScope - training environment check")
    print("")

    try:
        import torch
    except ImportError:
        print("PyTorch is NOT installed in this environment.")
        print("")
        print("  conda env update -f environment.yml   # adds pytorch to z-app")
        print("")
        print("See docs/rtmdet.md for the macOS/MPS notes and the pip fallback.")
        return 1

    print(f"python           {sys.version.split()[0]}")
    print(f"torch            {torch.__version__}")
    print(f"device count     cpu={1} mps={int(torch.backends.mps.is_available())}")
    mps_built = torch.backends.mps.is_built()
    print(f"mps built        {mps_built}")
    print(f"mps available    {torch.backends.mps.is_available()}")

    try:
        import cv2

        print(f"opencv           {cv2.__version__}")
    except ImportError as error:
        print(f"opencv           MISSING ({error}) - frames cannot be read")
        return 1

    try:
        import numpy

        print(f"numpy            {numpy.__version__}")
    except ImportError as error:
        print(f"numpy            MISSING ({error})")
        return 1

    from daphnia.vision.rtmdet import torch_status
    from daphnia.vision.rtmdet.data import CLASSES
    from daphnia.vision.rtmdet.model import build_model
    from daphnia.vision.rtmdet.train import pick_device

    print("")
    print(torch_status())

    device = pick_device()
    print("")
    print(f"building rtmdet-tiny for {len(CLASSES)} class(es) on {device} ...")
    model = build_model("tiny", len(CLASSES)).to(device)
    print(f"parameters       {model.parameter_count():,}")

    # A real forward *and* backward: that is what proves the accelerator works.
    sample = torch.zeros((1, 3, 320, 320), dtype=torch.float32, device=device)
    targets = [{"boxes": torch.tensor([[10.0, 10.0, 80.0, 70.0]], device=device),
                "labels": torch.tensor([0], device=device)}]
    outputs = model.loss(sample, targets)
    outputs["loss"].backward()
    print(f"forward+backward  ok   loss {float(outputs['loss']):.4f}   "
          f"({outputs['matched']} prior(s) matched)")

    trained = sum(1 for parameter in model.parameters() if parameter.grad is not None)
    print(f"gradients        {trained} tensor(s) received a gradient")

    from daphnia.vision.rtmdet.detect import default_checkpoint

    checkpoint = default_checkpoint()
    print("")
    print(f"checkpoint       {checkpoint}"
          + ("" if checkpoint.exists() else "   (not trained yet)"))
    print("")
    print("Ready. Train with:  make train   (or python scripts/train_rtmdet.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
