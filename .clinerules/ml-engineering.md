# Machine learning & deep learning — invoke the expert stance

**Trigger.** The task touches a model, a dataset that feeds a model, a loss, a metric,
training, inference, a checkpoint, model performance/accuracy/latency claims, or any file
under `daphnia/vision/rtmdet/**`, `daphnia/vision/keypoints.py`,
`daphnia/vision/detection.py`, `scripts/train_*`, `scripts/check_torch.py`,
`models/**` or `datasets/**`.

When the trigger fires, do all five, every time:

1. **Read `memory-bank/mlContext.md` in full** before proposing anything. It is part of
   the memory bank and it is not optional reading.
2. **Name the sub-role you are invoking** in the response — detection/pose expert,
   evaluation expert, data-pipeline expert, performance expert, or reproducibility
   expert — and let that role's standard govern the answer.
3. **Cite at least one reference implementation** from `mlContext.md` §5, and say what
   you took from it. **A design with no reference is a guess.** If nothing fits, say so
   explicitly and explain what you are borrowing from instead (a neighbouring task, a
   paper, the repo's own conventions).
4. **Ground the answer in this repo's ML discipline**: the ten laws (§2), the code shape
   (§3), and the test set that defines "done" (§4).
5. **Report with provenance** (§8) — corpus, split, threshold, epoch, device — and state
   what is *not* measured.

## Red lines (never do these, even if asked casually)

* **Never say "it should work".** Measure it, or say what remains unmeasured.
* **Never report a metric whose frame of reference is unstated** — and never call a
  hand-rolled IoU-0.5 score `mAP`.
* **Never introduce a silent fallback** (device, checkpoint, threshold, empty result).
  Fail with a message that names the file and the command that fixes it.
* **Never copy code from an AGPL project** (ultralytics YOLO). Ideas yes, code no.
* **Never present a paper's or another project's number as this project's result.**
* **Never write a model before the metric, the split and the data contract are stated.**
* **Never add a dependency without a licence check** and without saying why the
  standard library or the existing stack cannot do it.

## Where the detail lives

| Question | Read |
| --- | --- |
| How is ML code shaped here? What tests define "done"? | `memory-bank/mlContext.md` §3, §4 |
| Which repo should I read for this task, and what for? | `memory-bank/mlContext.md` §5 |
| What has already gone wrong, and what is the rule? | `memory-bank/mlContext.md` §6 |
| How do I train/run *this* detector? What do the metrics mean? | `docs/rtmdet.md` |
| Why is the corpus split the way it is? | `docs/requirements/SR-DATASET.md`, `SR-PREDICT.md` |
| Which surfaces consume a model, and how? | `memory-bank/systemPatterns.md` |
