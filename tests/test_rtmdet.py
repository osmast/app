"""RTMDet: the model, its losses, its data pipeline, and loading a checkpoint.

Everything here needs PyTorch, so a machine without it *skips* these tests rather
than failing the suite - the Qt application must stay testable without the ML stack
(``SR-PREDICT-N02``). ``tests/test_export.py`` covers the part that never needs
torch.
"""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from daphnia.vision.rtmdet import CLASSES, torch_available, torch_status

TORCH = torch_available()
requires_torch = unittest.skipUnless(TORCH, "PyTorch is not installed")

if TORCH:
    import torch

    from daphnia.vision.export import Export
    from daphnia.vision.rtmdet.data import (
        DetectionDataset,
        Letterbox,
        collate,
        horizontal_flip,
        letterbox,
        to_tensor,
    )
    from daphnia.vision.rtmdet.losses import (
        GIoULoss,
        QualityFocalLoss,
        SimOTAAssigner,
        batched_nms,
        bbox_overlaps,
        distance2bbox,
    )
    from daphnia.vision.rtmdet.model import build_model
    from daphnia.vision.rtmdet.train import (
        TrainConfig,
        checkpoint_payload,
        load_checkpoint,
        pick_device,
        save_checkpoint,
    )

#: A small input so the tests stay quick: still a multiple of 32.
SIZE = 128


def frame_with_boxes(boxes, width=SIZE * 2, height=SIZE):
    """A synthetic BGR frame: a pale field with dark rectangles on it.

    That is what a daphnid looks like to a detector - a small dark blob on a pale
    background - so the tests exercise the same contrast the real corpus has.
    """
    frame = np.full((height, width, 3), 220, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        frame[int(y1):int(y2), int(x1):int(x2)] = 40
    return frame


@requires_torch
class LossTests(unittest.TestCase):
    """The coding, the overlap, the NMS and the two losses."""

    def test_distance_coding_round_trips(self):
        points = torch.tensor([[5.0, 7.0]])
        boxes = torch.tensor([[1.0, 3.0, 11.0, 17.0]])
        distance = torch.tensor([[4.0, 4.0, 6.0, 10.0]])
        self.assertTrue(torch.allclose(distance2bbox(points, distance), boxes))
        self.assertTrue(torch.allclose(
            bbox_overlaps(distance2bbox(points, distance), boxes).diagonal(),
            torch.ones(1)))

    def test_overlap_of_identical_boxes_is_one(self):
        box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        self.assertAlmostEqual(float(bbox_overlaps(box, box)[0, 0]), 1.0, places=5)

    def test_overlap_of_disjoint_boxes_is_zero_and_giou_goes_negative(self):
        first = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        second = torch.tensor([[50.0, 50.0, 60.0, 60.0]])
        self.assertAlmostEqual(float(bbox_overlaps(first, second)[0, 0]), 0.0)
        self.assertLess(float(bbox_overlaps(first, second, mode="giou")[0, 0]), 0.0)

    def test_nms_keeps_the_best_of_two_overlapping_boxes(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]],
                         dtype=np.float32)
        keep = batched_nms(boxes, np.array([0.9, 0.8, 0.7]), np.zeros(3),
                           iou_threshold=0.5)
        self.assertEqual(keep, [0, 2])

    def test_nms_is_class_aware(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
        keep = batched_nms(boxes, np.array([0.9, 0.8]), np.array([0, 1]),
                           iou_threshold=0.5)
        self.assertEqual(keep, [0, 1])

    def test_nms_respects_the_score_threshold(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        self.assertEqual(batched_nms(boxes, np.array([0.2]), np.zeros(1),
                                     score_threshold=0.5), [])

    def test_quality_focal_loss_prefers_the_soft_target(self):
        loss = QualityFocalLoss()
        target = torch.tensor([[0.8]])
        good = loss(torch.tensor([[2.0]]), target)
        bad = loss(torch.tensor([[-2.0]]), target)
        self.assertLess(float(good), float(bad))

    def test_giou_loss_is_zero_for_a_perfect_box(self):
        loss = GIoULoss()
        box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        self.assertAlmostEqual(
            float(loss(box, box, torch.ones(1))), 0.0, places=5)

    def test_giou_loss_ignores_the_rows_a_weight_of_zero_marks(self):
        """A degenerate background prediction must not reach the mean at all."""
        loss = GIoULoss()
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 5.0, 5.0]])
        target = torch.tensor([[0.0, 0.0, 10.0, 10.0], [9.0, 9.0, 9.0, 9.0]])
        weight = torch.tensor([1.0, 0.0])
        self.assertAlmostEqual(float(loss(pred, target, weight)), 0.0, places=5)


@requires_torch
class AssignerTests(unittest.TestCase):
    """SimOTA: which locations become positives, and what they are told."""

    def _priors(self, rows=4, columns=4, stride=16):
        priors = torch.zeros((rows * columns, 4), dtype=torch.float32)
        for row in range(rows):
            for column in range(columns):
                priors[row * columns + column] = torch.tensor(
                    [(column + 0.5) * stride, (row + 0.5) * stride,
                     float(stride), float(stride)])
        return priors

    def test_the_location_under_an_animal_becomes_positive(self):
        assigner = SimOTAAssigner(num_classes=1)
        priors = self._priors()
        stride = 16.0
        boxes = distance2bbox(priors[:, :2], torch.full((16, 4), stride / 4))
        ground_truth = torch.tensor(
            [[1.5 * stride, 1.5 * stride, 2.5 * stride, 2.5 * stride]])
        _, _, box_target, box_weight, matched = assigner.assign(
            priors, torch.zeros((16, 1)), boxes, ground_truth,
            torch.zeros(1, dtype=torch.long))
        self.assertGreaterEqual(matched, 1)
        self.assertEqual(int(box_weight.sum()), matched)
        for index in torch.nonzero(box_weight, as_tuple=False).flatten():
            self.assertTrue(torch.allclose(box_target[index], ground_truth[0]))

    def test_the_soft_label_is_the_iou_of_the_assigned_box(self):
        """The classification target *is* the quality of the assigned prediction."""
        assigner = SimOTAAssigner(num_classes=1)
        priors = self._priors()
        stride = 16.0
        boxes = distance2bbox(priors[:, :2], torch.full((16, 4), stride / 2))
        ground_truth = torch.tensor(
            [[1.5 * stride, 1.5 * stride, 2.5 * stride, 2.5 * stride]])
        cls_target, _, _, box_weight, matched = assigner.assign(
            priors, torch.zeros((16, 1)), boxes, ground_truth,
            torch.zeros(1, dtype=torch.long))
        self.assertGreaterEqual(matched, 1)
        positive = torch.nonzero(box_weight, as_tuple=False).flatten()
        for row in positive:
            expected = float(bbox_overlaps(
                boxes[row].reshape(1, 4), ground_truth)[0, 0])
            self.assertAlmostEqual(float(cls_target[row, 0]), expected, places=5)
            self.assertLessEqual(float(cls_target[row, 0]), 1.0)

    def test_a_clear_animal_gets_a_high_quality_label(self):
        """A big, well-covered animal must not be labelled like a marginal one."""
        assigner = SimOTAAssigner(num_classes=1)
        priors = self._priors()
        stride = 16.0
        # Predictions the size of the animal, centred on their prior.
        boxes = distance2bbox(priors[:, :2], torch.full((16, 4), stride))
        ground_truth = torch.tensor(
            [[1.0 * stride, 1.0 * stride, 3.0 * stride, 3.0 * stride]])
        cls_target, _, _, box_weight, matched = assigner.assign(
            priors, torch.zeros((16, 1)), boxes, ground_truth,
            torch.zeros(1, dtype=torch.long))
        self.assertGreaterEqual(matched, 1)
        best = float(cls_target[box_weight > 0].max())
        self.assertGreater(best, 0.3)
        self.assertLess(best, 1.0)

    def test_an_empty_frame_assigns_nothing_but_keeps_classification_on(self):
        assigner = SimOTAAssigner(num_classes=1)
        priors = self._priors()
        _, cls_weight, _, box_weight, matched = assigner.assign(
            priors, torch.zeros((16, 1)), torch.zeros((16, 4)),
            torch.zeros((0, 4)), torch.zeros(0, dtype=torch.long))
        self.assertEqual(matched, 0)
        self.assertEqual(float(box_weight.sum()), 0.0)
        # Every location is still taught "this is background".
        self.assertEqual(float(cls_weight.sum()), 16.0)

    def test_one_location_is_never_claimed_by_two_animals(self):
        """Overlapping animals must not share a positive location."""
        assigner = SimOTAAssigner(num_classes=1)
        priors = self._priors()
        stride = 16.0
        boxes = distance2bbox(priors[:, :2], torch.full((16, 4), stride / 2))
        ground_truth = torch.tensor([
            [1.0 * stride, 1.0 * stride, 3.0 * stride, 3.0 * stride],
            [1.2 * stride, 1.2 * stride, 2.8 * stride, 2.8 * stride]])
        _, _, _, box_weight, matched = assigner.assign(
            priors, torch.zeros((16, 1)), boxes, ground_truth,
            torch.tensor([0, 0], dtype=torch.long))
        self.assertEqual(int(box_weight.sum()), matched)      # one row per match
        self.assertLessEqual(matched, 16)


@requires_torch
class GeometryTests(unittest.TestCase):
    """The letterbox, the normalisation and the flip - the pixel bookkeeping."""

    def test_letterbox_pads_to_a_multiple_of_the_divisor(self):
        frame = np.zeros((90, 320, 3), dtype=np.uint8)
        padded, mapping = letterbox(frame, SIZE)
        self.assertEqual(padded.shape[0] % 32, 0)
        self.assertEqual(padded.shape[1] % 32, 0)
        self.assertEqual((mapping.source_width, mapping.source_height), (320, 90))

    def test_letterbox_without_a_size_only_pads(self):
        frame = np.zeros((90, 100, 3), dtype=np.uint8)
        padded, mapping = letterbox(frame, None)
        self.assertEqual(padded.shape[:2], (96, 128))
        self.assertEqual(mapping.scale, 1.0)

    def test_the_letterbox_is_invertible_at_the_corners(self):
        frame = np.zeros((90, 320, 3), dtype=np.uint8)
        _, mapping = letterbox(frame, SIZE)
        original = np.array([[0, 0, 320, 90], [10, 10, 40, 40]], dtype=np.float32)
        padded = original * mapping.scale
        back = mapping.invert(padded)
        self.assertTrue(np.allclose(back, original, atol=1.0))

    def test_the_flip_mirrors_the_boxes_with_the_frame(self):
        frame = frame_with_boxes([(10, 10, 40, 40)])
        boxes = np.array([[10, 10, 40, 40]], dtype=np.float32)
        flipped, mirrored = horizontal_flip(frame, boxes)
        self.assertTrue(np.allclose(flipped[:, 0], frame[:, -1]))
        self.assertAlmostEqual(float(mirrored[0][0]), SIZE * 2 - 40)
        self.assertAlmostEqual(float(mirrored[0][2]), SIZE * 2 - 10)

    def test_to_tensor_normalises_bgr_channels_first(self):
        frame = frame_with_boxes([])
        tensor = to_tensor(frame)
        self.assertEqual(tuple(tensor.shape), (3, SIZE, SIZE * 2))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertNotEqual(float(tensor.mean()), 0.0)


@requires_torch
class ModelTests(unittest.TestCase):
    """The network: shapes, one training step, and inference in image pixels."""

    def _model(self, preset="tiny"):
        return build_model(preset, len(CLASSES))

    def test_the_backbone_returns_three_levels_at_8_16_and_32(self):
        model = self._model()
        features = model.backbone(torch.zeros(1, 3, SIZE, SIZE))
        self.assertEqual([tuple(f.shape[-2:]) for f in features],
                         [(SIZE // 8, SIZE // 8), (SIZE // 16, SIZE // 16),
                          (SIZE // 32, SIZE // 32)])

    def test_each_stage_sees_the_previous_stage_not_the_stem(self):
        """The bug this guards: a chained backbone must actually be chained."""
        model = self._model()
        features = model.backbone(torch.zeros(1, 3, SIZE, SIZE))
        self.assertEqual([f.shape[1] for f in features],
                         list(model.backbone.out_channels))

    def test_the_head_shares_convolutions_and_keeps_a_batchnorm_per_level(self):
        model = self._model()
        self.assertEqual(len(model.head.shared), 2)          # stacked convs
        for index in range(len(model.head.shared)):
            self.assertEqual(len(model.head.bns[index]), 3)  # one per level

    def test_forward_returns_one_prediction_map_per_level(self):
        model = self._model()
        _, (cls_scores, bbox_preds) = model(torch.zeros(1, 3, SIZE, SIZE))
        self.assertEqual(len(cls_scores), 3)
        self.assertEqual(len(bbox_preds), 3)
        self.assertEqual(cls_scores[0].shape[1], len(CLASSES))
        self.assertEqual(bbox_preds[0].shape[1], 4)

    def test_a_training_step_produces_a_finite_loss_and_gradients(self):
        model = self._model()
        images = torch.zeros(1, 3, SIZE, SIZE)
        targets = [{"boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0]]),
                    "labels": torch.tensor([0])}]
        outputs = model.loss(images, targets)
        self.assertTrue(bool(torch.isfinite(outputs["loss"])))
        outputs["loss"].backward()
        graded = sum(1 for p in model.parameters() if p.grad is not None)
        self.assertGreater(graded, 50)
        self.assertGreaterEqual(outputs["matched"], 1)

    def test_an_empty_frame_still_produces_a_finite_loss(self):
        """A frame with no animal is a legitimate sample: it teaches background."""
        model = self._model()
        outputs = model.loss(torch.zeros(1, 3, SIZE, SIZE),
                             [{"boxes": torch.zeros((0, 4)),
                               "labels": torch.zeros(0, dtype=torch.long)}])
        self.assertTrue(bool(torch.isfinite(outputs["loss"])))
        self.assertEqual(outputs["matched"], 0)

    def test_the_model_starts_by_saying_background(self):
        """The 0.01 class prior: an untrained head must not be confident."""
        model = self._model().eval()
        with torch.no_grad():
            _, (cls_scores, _) = model(torch.zeros(1, 3, SIZE, SIZE))
        self.assertLess(float(cls_scores[0].sigmoid().max()), 0.05)

    def test_detect_returns_boxes_inside_the_image(self):
        model = self._model().eval()
        with torch.no_grad():
            results = model.detect(torch.zeros(1, 3, SIZE, SIZE),
                                   score_threshold=0.0)
        boxes, scores, labels = results[0]
        self.assertEqual(len(boxes), len(scores))
        self.assertEqual(len(boxes), len(labels))
        for box in boxes:
            self.assertGreaterEqual(float(box[0]), 0.0)
            self.assertGreaterEqual(float(box[1]), 0.0)
            self.assertLessEqual(float(box[2]), float(SIZE))
            self.assertLessEqual(float(box[3]), float(SIZE))

    def test_the_presets_differ_in_size(self):
        self.assertLess(self._model("tiny").parameter_count(),
                        self._model("small").parameter_count())

    def test_an_unknown_preset_is_refused(self):
        with self.assertRaises(ValueError):
            build_model("enormous", 1)


@requires_torch
class CheckpointTests(unittest.TestCase):
    """A checkpoint must be loadable by a reader that has only the file."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "checkpoint.pt"

    def test_a_checkpoint_round_trips_and_remembers_how_it_was_built(self):
        model = build_model("tiny", len(CLASSES))
        config = TrainConfig(input_size=320)
        save_checkpoint(self.path, checkpoint_payload(
            model, config, None, {"ap50": 0.5}, {"boxes": 3}))
        loaded, payload = load_checkpoint(self.path)
        self.assertEqual(payload["preset"], "tiny")
        self.assertEqual(payload["input_size"], 320)
        self.assertEqual(payload["classes"], list(CLASSES))
        self.assertEqual(payload["metrics"]["ap50"], 0.5)
        self.assertEqual(loaded.parameter_count(), model.parameter_count())

    def test_saving_kept_weights_writes_those_weights_not_the_current_ones(self):
        """The metrics in a checkpoint must describe the tensors beside them."""
        model = build_model("tiny", len(CLASSES))
        kept = {key: torch.zeros_like(value)
                for key, value in model.state_dict().items()}
        payload = checkpoint_payload(model, TrainConfig(), None, {"ap50": 0.9},
                                     weights=kept)
        self.assertTrue(all(float(value.abs().sum()) == 0.0
                            for value in payload["model"].values()))
        self.assertEqual(payload["metrics"]["ap50"], 0.9)

    def test_saving_is_atomic_so_no_half_written_file_is_left(self):
        model = build_model("tiny", len(CLASSES))
        save_checkpoint(self.path, checkpoint_payload(model, TrainConfig()))
        self.assertFalse(self.path.with_suffix(".pt.part").exists())

    def test_a_missing_checkpoint_is_named(self):
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(self.path)

    def test_a_foreign_checkpoint_is_refused_with_its_reason(self):
        torch.save({"something": "else"}, self.path)
        with self.assertRaises(ValueError) as caught:
            load_checkpoint(self.path)
        self.assertIn("not a DaphniaScope RTMDet checkpoint",
                      str(caught.exception))

    def test_the_device_is_chosen_for_this_machine(self):
        self.assertIn(pick_device("auto").type, ("mps", "cpu"))
        self.assertEqual(pick_device("cpu").type, "cpu")

    def test_asking_for_mps_when_it_is_missing_is_an_error(self):
        if torch.backends.mps.is_available():
            self.skipTest("this machine has MPS")
        with self.assertRaises(RuntimeError):
            pick_device("mps")


@requires_torch
class DataPipelineTests(unittest.TestCase):
    """Frames, boxes and a batch - including the mosaic a small corpus needs."""

    def setUp(self):
        from tests.test_export import specimen, write_export

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        folder = Path(self._tmp.name)
        records = []
        for index in range(6):
            image = f"images/f{index}.png"
            records.append(specimen(image, 10, 10, 60, 60, split="train",
                                    capture=f"/project/OBS_{index}"))
        write_export(folder, records)
        # Real pixels stand in for the export's placeholder bytes.
        import cv2

        for index in range(6):
            cv2.imwrite(str(folder / f"images/f{index}.png"),
                        frame_with_boxes([(10, 10, 60, 60)]))
        from daphnia.vision.export import Export

        self.export = Export.read(folder)

    def test_the_dataset_finds_the_frames_of_a_split(self):
        dataset = DetectionDataset(self.export, "train", input_size=SIZE)
        self.assertEqual(len(dataset), 6)

    def test_a_sample_is_a_letterboxed_frame_with_its_boxes(self):
        dataset = DetectionDataset(self.export, "train", input_size=SIZE)
        sample = dataset[0]
        self.assertEqual(sample.image.shape[0] % 32, 0)
        self.assertEqual(len(sample.boxes), 1)
        self.assertEqual(len(sample.labels), 1)

    def test_augmented_samples_stay_valid_and_reproducible(self):
        dataset = DetectionDataset(self.export, "train", input_size=SIZE,
                                   augment=True, seed=7)
        first = dataset[0]
        again = dataset[0]
        self.assertTrue(np.allclose(first.boxes, again.boxes))
        dataset.set_epoch(1)
        for index in range(len(dataset)):
            sample = dataset[index]
            self.assertEqual(sample.image.shape[:2],
                             (sample.letterbox.target_height,
                              sample.letterbox.target_width))
            for x1, y1, x2, y2 in sample.boxes:
                self.assertGreaterEqual(x1, 0.0)
                self.assertGreaterEqual(y1, 0.0)
                self.assertLessEqual(x2, sample.letterbox.target_width)
                self.assertLessEqual(y2, sample.letterbox.target_height)

    def test_collate_stacks_a_batch_and_keeps_the_sample_list(self):
        dataset = DetectionDataset(self.export, "train", input_size=SIZE)
        batch = collate([dataset[0], dataset[1], dataset[2]])
        height, width = dataset[0].image.shape[:2]
        self.assertEqual(tuple(batch["images"].shape), (3, 3, height, width))
        self.assertEqual(len(batch["targets"]), 3)
        self.assertEqual(len(batch["samples"]), 3)

    def test_the_class_name_is_the_organism_not_an_index(self):
        dataset = DetectionDataset(self.export, "train", input_size=SIZE)
        self.assertEqual(dataset.class_names(), CLASSES)


@requires_torch
class EvaluationTests(unittest.TestCase):
    """Matching happens in ONE frame of reference, or the score is a lie."""

    def setUp(self):
        from tests.test_export import specimen, write_export

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        folder = Path(self._tmp.name)
        # A deliberately non-square frame, so the letterbox actually rescales: on a
        # square input a padded box and an original box differ by a pure padding,
        # which is easier to get right by accident.
        records = [specimen(f"images/f{index}.png", 200, 100, 500, 320,
                            split="val", capture=f"/project/OBS_{index}")
                   for index in range(3)]
        write_export(folder, records)
        import cv2

        for index in range(3):
            cv2.imwrite(str(folder / f"images/f{index}.png"),
                        frame_with_boxes([(200, 100, 500, 320)], width=1280,
                                         height=720))
        self.export = Export.read(folder)

    class StubModel:
        """The minimum :func:`evaluate` needs of a detector: a mode and `detect`."""

        training = False

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def eval(self):
            self.training = False
            return self

    def test_a_perfect_detector_scores_a_perfect_recall_on_a_letterboxed_frame(self):
        from daphnia.vision.rtmdet.train import TrainConfig, evaluate

        dataset = DetectionDataset(self.export, "val", input_size=SIZE,
                                   augment=False)
        truth = dataset[0].boxes
        stub = self.StubModel

        class Oracle(stub):
            """A flawless detector: it emits the labels, in the padded input frame.

            That is exactly what a perfect model would return, so an honest
            evaluation must score it 1.0. Any other number means the two sides were
            compared in different coordinate frames - which is what happened.
            """

            def detect(self, images, score_threshold=0.05, iou_threshold=0.6):
                return [(np.asarray(truth, dtype=np.float32),
                         np.ones(len(truth), dtype=np.float32),
                         np.zeros(len(truth), dtype=np.int64))]

        metrics = evaluate(Oracle(), dataset, TrainConfig(input_size=SIZE),
                           device="cpu")
        self.assertEqual(metrics["true_positives"], len(dataset) * len(truth))
        self.assertEqual(metrics["false_positives"], 0)
        self.assertEqual(metrics["false_negatives"], 0)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["ap50"], 1.0)

    def test_a_detector_that_misses_everything_scores_zero(self):
        from daphnia.vision.rtmdet.train import TrainConfig, evaluate

        dataset = DetectionDataset(self.export, "val", input_size=SIZE,
                                   augment=False)
        stub = self.StubModel

        class Blind(stub):
            def detect(self, images, score_threshold=0.05, iou_threshold=0.6):
                return [(np.zeros((0, 4), dtype=np.float32),
                         np.zeros(0, dtype=np.float32),
                         np.zeros(0, dtype=np.int64))]

        metrics = evaluate(Blind(), dataset, TrainConfig(input_size=SIZE),
                           device="cpu")
        self.assertEqual(metrics["true_positives"], 0)
        self.assertEqual(metrics["detections"], 0)
        self.assertGreater(metrics["false_negatives"], 0)
        self.assertAlmostEqual(metrics["ap50"], 0.0)


class EnvironmentReportingTests(unittest.TestCase):
    """What a surface can say about PyTorch without importing it for real."""

    def test_the_status_line_always_says_something(self):
        self.assertTrue(torch_status())

    def test_the_status_line_names_the_device_when_torch_is_there(self):
        if not TORCH:
            self.assertIn("not installed", torch_status())
            return
        self.assertIn("PyTorch", torch_status())
        self.assertIn("device", torch_status())


if __name__ == "__main__":
    unittest.main()


