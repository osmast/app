"""RTMPose: the crop mapping, the SimCC maths, the model, the metric, the checkpoint.

Everything here needs PyTorch, so a machine without it *skips* these tests rather
than failing the suite - the Qt application must stay testable without the ML stack
(``SR-PREDICT-N02``). ``tests/test_pose_corpus.py`` covers the part that never needs
torch.

The two tests that matter most are the ones this project has already been burned by
without: the **round trip** of the box-to-input mapping (``mlContext.md`` law 1 - a
coordinate bug that reads 0.000 for a model that is learning) and the **oracle** of
the metric (law 2 - a perfect prediction must score 1.0, a blind one 0.0).
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from daphnia.vision.rtmpose import torch_available

TORCH = torch_available()
requires_torch = unittest.skipUnless(TORCH, "PyTorch is not installed")

if TORCH:
    import cv2
    import torch

    from daphnia.vision.pose_corpus import PoseCorpus
    from daphnia.vision.rtmpose.data import (
        PoseDataError,
        PoseDataset,
        PoseInput,
        collate,
        pose_input_for_box,
        window_image,
    )
    from daphnia.vision.rtmpose.losses import (
        gaussian_1d,
        keypoint_errors,
        oks,
        pck,
        simcc_decode,
        simcc_encode,
        simcc_loss,
    )
    from daphnia.vision.rtmpose.model import build_model
    from daphnia.vision.rtmpose.train import (
        TrainConfig,
        checkpoint_payload,
        load_checkpoint,
        pose_metrics,
        save_checkpoint,
    )

#: A small square so the tests stay quick - still a multiple of 8.
SIZE = 64

SCHEMA = {"key": "pose_head_tail", "keypoints": ["head", "spine base"],
          "export_names": ["head", "tail"]}


def _write_png(path, width=160, height=120, box=(20, 20, 60, 60)):
    """A pale frame with one dark rectangle - enough to crop and resize."""
    frame = np.full((height, width, 3), 200, np.uint8)
    x1, y1, x2, y2 = box
    frame[y1:y2, x1:x2] = 40
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)
    return frame


def make_corpus(folder, captures=("b01", "b02", "b03", "b04")):
    """A tiny real corpus on disk: PNGs plus a labels.json with head/tail."""
    specimens = []
    for capture in captures:
        for frame_index in range(2):
            stem = f"{capture}_f{frame_index:02d}_00000001"
            relative = f"images/{stem}_deadbeef.png"
            _write_png(folder / relative)
            specimens.append({
                "file": f"/original/{capture}/{stem}.png",
                "image": relative, "split": "train",
                "image_width": 160, "image_height": 120, "specimen_index": 1,
                "keypoints": {"head": [30.0, 30.0], "tail": [55.0, 50.0]},
                "traits": {"length": {"length_px": 35.0, "length_um": 2600.0}},
                "bbox": [20.0, 20.0, 60.0, 60.0]})
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "labels.json").write_text(json.dumps(
        {"dataset": {"name": "t"}, "schema": dict(SCHEMA), "specimens": specimens}),
        encoding="utf-8")
    return folder


@requires_torch
class PoseInputMappingTests(unittest.TestCase):
    """Input pixels and source pixels, in both directions - hand-computed."""

    def test_a_point_is_converted_between_the_two_frames(self):
        pose = PoseInput(origin=(100, 50), side=200, size=100)
        self.assertEqual(pose.to_input((150, 100)), (25.0, 25.0))
        self.assertEqual(pose.invert((25.0, 25.0)), (150.0, 100.0))

    def test_the_round_trip_is_lossless(self):
        pose = PoseInput(origin=(496, 539), side=139, size=64)
        for point in ((500.5, 540.25), (600.0, 630.0), (634.75, 660.5)):
            self.assertEqual(pose.invert(pose.to_input(point)), point)

    def test_the_origin_and_side_are_integers(self):
        pose = PoseInput(origin=(10.4, 20.6), side=199.5, size=64)
        self.assertEqual(pose.origin, (10, 21))
        self.assertEqual(pose.side, 200)
        self.assertEqual(pose.scale, 64 / 200)

    def test_contains_tells_which_source_points_are_inside(self):
        pose = PoseInput(origin=(100, 50), side=40, size=20)
        self.assertTrue(pose.contains((100, 50)))
        self.assertTrue(pose.contains((139, 89)))
        self.assertFalse(pose.contains((99, 50)))
        self.assertFalse(pose.contains((140, 89)))


@requires_torch
class PoseWindowTests(unittest.TestCase):
    """The window geometry: square, centred, and invertible in closed form."""

    def test_the_window_side_is_the_box_side_times_one_plus_two_margins(self):
        pose = pose_input_for_box((20, 20, 60, 60), (160, 120), size=SIZE, margin=0.25)
        self.assertEqual(pose.side, 60)                     # 40 * 1.5
        self.assertEqual(pose.origin, (10, 10))             # centred on (40, 40)

    def test_the_window_contains_the_whole_box(self):
        box = (100, 40, 140, 60)
        pose = pose_input_for_box(box, (300, 200), size=SIZE, margin=0.25)
        for corner in ((100, 40), (140, 60)):
            self.assertTrue(pose.contains(corner))

    def test_a_degenerate_box_is_refused(self):
        with self.assertRaises(PoseDataError):
            pose_input_for_box((10, 10, 10, 40), (160, 120))

    def test_the_window_image_is_the_input_square(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        frame[20:60, 20:60] = 255
        pose = pose_input_for_box((20, 20, 60, 60), (160, 120), size=SIZE)
        window = window_image(frame, pose)
        self.assertEqual(window.shape[:2], (SIZE, SIZE))

    def test_a_window_past_the_edge_is_edge_padded_not_black(self):
        frame = np.full((120, 160, 3), 200, np.uint8)
        pose = PoseInput(origin=(-20, -20), side=160, size=SIZE)
        window = window_image(frame, pose)
        self.assertEqual(window.shape[:2], (SIZE, SIZE))
        self.assertTrue(np.all(window == 200))              # the edge value, not 0

    def test_a_window_that_misses_the_frame_returns_nothing(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        pose = PoseInput(origin=(500, 500), side=40, size=SIZE)
        self.assertIsNone(window_image(frame, pose))
        self.assertIsNone(window_image(None, pose))


@requires_torch
class SimCCTests(unittest.TestCase):
    """Coding a point to a distribution and decoding it back - hand-computed."""

    def test_gaussian_1d_is_a_normalised_distribution(self):
        weights = gaussian_1d(5.0, 11, sigma=2.0)
        self.assertEqual(weights.shape, (11,))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)
        self.assertEqual(int(np.argmax(weights)), 5)

    def test_encode_decode_round_trip_is_exact_on_the_grid(self):
        split_ratio = 2.0
        points = np.array([[10.0, 20.0], [31.5, 47.0]], np.float32)
        x_target, y_target = simcc_encode(points, (SIZE, SIZE), split_ratio)
        x_logits = torch.from_numpy(np.log(x_target + 1e-12))[None]
        y_logits = torch.from_numpy(np.log(y_target + 1e-12))[None]
        decoded = simcc_decode(x_logits, y_logits, split_ratio)[0].numpy()
        self.assertTrue(np.allclose(decoded, points, atol=0.5), decoded)

    def test_the_bin_count_matches_the_split_ratio(self):
        x_target, _ = simcc_encode(np.array([[0.0, 0.0]], np.float32),
                                   (SIZE, SIZE), 2.0)
        self.assertEqual(x_target.shape[-1], int(round(SIZE * 2.0)) + 1)

    def test_a_correct_target_scores_lower_than_a_shifted_one(self):
        points = np.array([[20.0, 20.0]], np.float32)
        x_target, y_target = simcc_encode(points, (SIZE, SIZE), 2.0)
        right_x = torch.from_numpy(np.log(x_target + 1e-12))[None]
        right_y = torch.from_numpy(np.log(y_target + 1e-12))[None]
        shifted_points = np.array([[50.0, 50.0]], np.float32)
        sx, sy = simcc_encode(shifted_points, (SIZE, SIZE), 2.0)
        wrong_x = torch.from_numpy(np.log(sx + 1e-12))[None]
        wrong_y = torch.from_numpy(np.log(sy + 1e-12))[None]
        good = float(simcc_loss(right_x, right_y, x_target, y_target)[0])
        bad = float(simcc_loss(wrong_x, wrong_y, x_target, y_target)[0])
        self.assertLess(good, bad)

    def test_the_loss_reports_both_axes(self):
        x_target, y_target = simcc_encode(np.array([[20.0, 30.0]], np.float32),
                                          (SIZE, SIZE), 2.0)
        logits = torch.zeros(1, 2, int(round(SIZE * 2.0)) + 1)
        total, loss_x, loss_y = simcc_loss(logits, logits, x_target, y_target)
        self.assertEqual(total.shape, torch.Size([]))
        self.assertGreaterEqual(float(loss_x), 0.0)
        self.assertGreaterEqual(float(loss_y), 0.0)


@requires_torch
class MetricTests(unittest.TestCase):
    """The oracle (law 2): a perfect prediction scores 1.0, a blind one 0.0."""

    def setUp(self):
        self.truth = np.array([[[10.0, 20.0], [60.0, 70.0]],
                               [[30.0, 40.0], [80.0, 90.0]]])
        self.areas = np.array([2500.0, 2500.0])

    def test_a_perfect_prediction_scores_perfectly(self):
        metrics = pose_metrics(self.truth, self.truth, areas=self.areas,
                               names=("head", "tail"), um_per_px=74.8)
        self.assertEqual(metrics["median_error_px"], 0.0)
        self.assertEqual(metrics["pck"]["1px"], 1.0)
        self.assertAlmostEqual(metrics["oks"], 1.0, places=6)
        self.assertEqual(metrics["length_median_error_um"], 0.0)

    def test_a_blind_prediction_scores_zero(self):
        blind = np.zeros_like(self.truth)
        metrics = pose_metrics(blind, self.truth, areas=self.areas,
                               names=("head", "tail"), um_per_px=74.8)
        self.assertEqual(metrics["pck"]["1px"], 0.0)
        self.assertLess(metrics["oks"], 1e-3)

    def test_a_shape_mismatch_is_a_frame_of_reference_bug(self):
        with self.assertRaises(ValueError) as caught:
            pose_metrics(self.truth[:1], self.truth)
        self.assertIn("frame-of-reference", str(caught.exception))

    def test_the_um_thresholds_are_restated(self):
        metrics = pose_metrics(self.truth, self.truth, areas=self.areas,
                               thresholds_px=(1.0,), um_per_px=74.8)
        self.assertIn("75um", metrics["pck_um"])

    def test_pck_and_oks_stand_alone(self):
        self.assertEqual(pck(np.array([0.5, 2.0]), 1.0), 0.5)
        self.assertEqual(pck(np.array([]), 1.0), 0.0)
        self.assertAlmostEqual(oks(self.truth[0], self.truth[0], 2500.0), 1.0, places=6)
        self.assertEqual(keypoint_errors(self.truth, self.truth).max(), 0.0)


@requires_torch
class ModelTests(unittest.TestCase):
    """The network's shapes, and its refusal to accept the wrong input size."""

    def setUp(self):
        self.model = build_model("tiny", 2, SIZE, 2.0, ("head", "tail"))

    def test_forward_returns_one_distribution_per_axis(self):
        x_logits, y_logits = self.model(torch.zeros(1, 3, SIZE, SIZE))
        bins = int(round(SIZE * 2.0)) + 1
        self.assertEqual(tuple(x_logits.shape), (1, 2, bins))
        self.assertEqual(tuple(y_logits.shape), (1, 2, bins))

    def test_predict_returns_points_in_input_pixels(self):
        points = self.model.predict(torch.zeros(2, 3, SIZE, SIZE))
        self.assertEqual(tuple(points.shape), (2, 2, 2))
        self.assertTrue(float(points.min()) >= 0.0)
        self.assertTrue(float(points.max()) <= SIZE)
        self.assertTrue(torch.isfinite(points).all())

    def test_the_wrong_input_size_is_a_named_error_not_a_guess(self):
        with self.assertRaises(ValueError) as caught:
            self.model(torch.zeros(1, 3, SIZE // 2, SIZE // 2))
        self.assertIn("law 3", str(caught.exception))

    def test_an_unknown_preset_is_refused(self):
        with self.assertRaises(ValueError):
            build_model("huge", 2, SIZE)


@requires_torch
class DatasetTests(unittest.TestCase):
    """One box becomes one square sample, deterministically and in the right frame."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.corpus = PoseCorpus.read(make_corpus(Path(self.tmp.name) / "kp"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_sample_is_the_input_square_with_keypoints_inside(self):
        dataset = PoseDataset(self.corpus, "train", input_size=SIZE)
        self.assertEqual(len(dataset), 8)
        sample = dataset[0]
        self.assertEqual(sample.image.shape[:2], (SIZE, SIZE))
        self.assertTrue(np.all(sample.keypoints >= -1.0))
        self.assertTrue(np.all(sample.keypoints <= SIZE + 1.0))
        self.assertTrue(np.all(sample.visible == 1.0))

    def test_the_augmentation_stream_is_reproducible(self):
        dataset = PoseDataset(self.corpus, "train", input_size=SIZE, augment=True,
                              seed=7)
        dataset.set_epoch(3)
        first, second = dataset[0], dataset[0]
        self.assertTrue(np.array_equal(first.image, second.image))
        self.assertTrue(np.allclose(first.keypoints, second.keypoints))

    def test_a_certain_flip_mirrors_the_keypoints(self):
        plain = PoseDataset(self.corpus, "train", input_size=SIZE, augment=False)[0]
        flipped = PoseDataset(self.corpus, "train", input_size=SIZE, augment=True,
                              seed=0, flip_probability=1.0)[0]
        self.assertTrue(np.allclose(flipped.keypoints[:, 0],
                                    SIZE - plain.keypoints[:, 0]))
        self.assertTrue(np.allclose(flipped.keypoints[:, 1], plain.keypoints[:, 1]))

    def test_collate_stacks_images_and_keeps_targets(self):
        dataset = PoseDataset(self.corpus, "train", input_size=SIZE)
        batch = collate([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["images"].shape), (2, 3, SIZE, SIZE))
        self.assertEqual(len(batch["targets"]), 2)
        self.assertEqual(tuple(batch["targets"][0]["keypoints"].shape), (2, 2))
        self.assertEqual(len(batch["samples"]), 2)

    def test_the_source_keypoints_round_trip_through_the_mapping(self):
        sample = PoseDataset(self.corpus, "train", input_size=SIZE)[0]
        truth = np.asarray([[x, y] for _name, x, y in sample.source_points])
        self.assertTrue(np.allclose(sample.source_keypoints(), truth, atol=1e-4))


@requires_torch
class CheckpointTests(unittest.TestCase):
    """A checkpoint rebuilds the model it describes, and refuses anything else."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = TrainConfig(input_size=SIZE)
        self.model = build_model("tiny", 2, SIZE, 2.0, ("head", "tail"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_saved_checkpoint_loads_and_predicts_identically(self):
        path = Path(self.tmp.name) / "pose.pt"
        save_checkpoint(path, checkpoint_payload(self.model, self.config))
        loaded, payload = load_checkpoint(path, device="cpu")
        self.assertEqual(payload["input_size"], SIZE)
        self.assertEqual(tuple(payload["keypoints"]), ("head", "tail"))
        images = torch.rand(1, 3, SIZE, SIZE)
        self.model.eval()               # BatchNorm uses batch stats while training,
        self.assertTrue(torch.allclose(  # so a like-for-like compare needs both modes
            self.model.predict(images), loaded.predict(images), atol=1e-5))

    def test_a_foreign_file_is_refused_with_its_path(self):
        path = Path(self.tmp.name) / "not.pt"
        path.write_bytes(b"not a checkpoint")
        with self.assertRaises(ValueError) as caught:
            load_checkpoint(path)
        self.assertIn("not.pt", str(caught.exception))

    def test_a_missing_checkpoint_names_the_file(self):
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(Path(self.tmp.name) / "absent.pt")


class PoseModelCropTests(unittest.TestCase):
    """The crop the model is *served* must be the crop it was *trained* with (law 3).

    The crop margin decides how far the square window reaches beyond the box, and therefore
    where a keypoint lands once it is mapped back to source pixels. A model trained with one
    margin and served with another places every keypoint systematically wrong - a scale error
    on the crop, not a visible crash - so the margin travels in the checkpoint and the
    inference object reads it, exactly as it reads the input size.
    """

    class _StubModel:
        """The attributes ``PoseModel`` reads off the model, and nothing else."""

        input_size = 192
        keypoint_names = ("head", "tail")

    def _pose_model(self, payload):
        from daphnia.vision.rtmpose.detect import PoseModel

        return PoseModel(self._StubModel(), payload, "cpu")

    def test_the_margin_is_read_from_the_checkpoint(self):
        from daphnia.vision.crop import DEFAULT_MARGIN

        self.assertEqual(self._pose_model({}).margin, DEFAULT_MARGIN)
        self.assertEqual(self._pose_model({"config": {"margin": 0.5}}).margin, 0.5)

    def test_a_foreign_margin_moves_the_window_the_keypoints_are_mapped_through(self):
        from daphnia.vision.crop import DEFAULT_MARGIN
        from daphnia.vision.rtmpose.data import pose_input_for_box

        box = (20, 20, 60, 60)
        as_trained = pose_input_for_box(box, (160, 120), 192, margin=DEFAULT_MARGIN)
        model = self._pose_model({"config": {"margin": 0.5}})
        as_served = pose_input_for_box(box, (160, 120), model.input_size,
                                       margin=model.margin)
        self.assertNotEqual(as_trained.side, as_served.side)


if __name__ == "__main__":
    unittest.main()
