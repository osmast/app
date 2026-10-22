"""The models in force, run on one frame - with stubs, so no PyTorch is needed.

What is tested here is the *wiring*: which checkpoint is used, how a detection is paired with
its pose, what the label says, and that nothing is stored. The models themselves are tested in
`tests/test_rtmdet.py` / `tests/test_rtmpose.py` (both `skipUnless(torch)`).
"""

import unittest
from dataclasses import dataclass

from daphnia.ui.model_pass import ModelRunner, Prediction, primary


@dataclass
class StubDetection:
    box: tuple
    score: float = 0.5


@dataclass
class StubPose:
    """A pose as `rtmpose` reports it: named keypoints, in the model's own order."""

    box: tuple
    points: dict
    names: tuple = ("head", "tail")

    def length_px(self) -> float:
        (x1, y1), (x2, y2) = self.points["head"], self.points["tail"]
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


class StubDetector:
    def __init__(self, detections):
        self._detections = list(detections)
        self.seen = []

    def detect(self, frame):
        self.seen.append(frame)
        return list(self._detections)

    def describe(self):
        return "stub-tiny   classes: daphnia"


class StubPoseModel:
    def __init__(self, poses):
        self._poses = list(poses)
        self.boxes_seen = None

    def detect_frame(self, frame, detections=None):
        self.boxes_seen = [tuple(item.box) for item in (detections or [])]
        return list(self._poses)


class RunnerTests(unittest.TestCase):
    """One pass: RTMDet's boxes, RTMPose's keypoints inside them, one line to show."""

    def test_one_pass_returns_a_prediction_per_detection(self):
        runner = ModelRunner()
        runner.detector = StubDetector([StubDetection((0, 0, 10, 10)),
                                        StubDetection((20, 20, 60, 60))])
        result = runner.run(object())
        self.assertEqual(len(result), 2)
        self.assertEqual([item.box for item in result.predictions],
                         [(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 60.0, 60.0)])
        self.assertIn("2 animal(s)", result.note)
        self.assertIn("ms", result.note)

    def test_a_pose_is_matched_by_its_box_never_by_position(self):
        """The pose pass skips boxes it cannot crop, so the two lists are not parallel."""
        runner = ModelRunner()
        runner.detector = StubDetector([StubDetection((0, 0, 10, 10)),
                                        StubDetection((20, 20, 60, 60))])
        runner.pose_model = StubPoseModel([StubPose((20, 20, 60, 60),
                                                    {"head": (30, 30), "tail": (50, 50)})])
        result = runner.run(object())
        self.assertEqual(result.predictions[0].points, ())
        self.assertEqual(result.predictions[1].points, ((30.0, 30.0), (50.0, 50.0)))
        self.assertIn("1 with keypoints", result.note)

    def test_the_label_is_um_only_with_a_scale(self):
        """Pixels and micrometres are never mixed (`SR-PREDICT-013`)."""
        runner = ModelRunner()
        runner.detector = StubDetector([StubDetection((0, 0, 10, 10))])
        runner.pose_model = StubPoseModel([StubPose((0, 0, 10, 10),
                                                    {"head": (0, 0), "tail": (0, 10)})])
        without = runner.run(object()).predictions[0]
        with_scale = runner.run(object(), um_per_px=2.0).predictions[0]
        self.assertEqual(without.label, "")
        self.assertEqual(with_scale.label, "20 um")

    def test_the_largest_animal_is_the_one_handed_to_the_draft(self):
        small = Prediction(box=(0, 0, 10, 10))
        large = Prediction(box=(0, 0, 100, 50))
        self.assertIs(primary([small, large]), large)
        self.assertIsNone(primary([]))

    def test_no_model_in_force_is_a_message_not_a_failure(self):
        runner = ModelRunner()
        runner.detector = None
        runner.load = lambda: False
        result = runner.run(object())
        self.assertEqual(len(result), 0)
        self.assertIn("No model in force", result.note)

    def test_a_failing_model_names_itself(self):
        class Broken:
            def detect(self, frame):
                raise RuntimeError("device exploded")

            def describe(self):
                return "broken"

        runner = ModelRunner()
        runner.detector = Broken()
        result = runner.run(object())
        self.assertEqual(len(result), 0)
        self.assertIn("RuntimeError", result.note)
        self.assertIn("device exploded", result.note)

    def test_no_frame_is_said_out_loud(self):
        runner = ModelRunner()
        runner.detector = StubDetector([])
        result = runner.run(None)
        self.assertIn("No frame", result.note)

    def test_list_frames_are_accepted_too(self):
        """A pose reader that reports a list of pairs works as well as a named mapping."""

        class ListPose:
            box = (0.0, 0.0, 10.0, 10.0)
            points = [(1, 1), (4, 5)]

        class ListPoseModel:
            def detect_frame(self, frame, detections=None):
                return [ListPose()]

        runner = ModelRunner()
        runner.detector = StubDetector([StubDetection((0, 0, 10, 10))])
        runner.pose_model = ListPoseModel()
        points = runner.run(object()).predictions[0].points
        self.assertEqual(points, ((1.0, 1.0), (4.0, 5.0)))


if __name__ == "__main__":
    unittest.main()
