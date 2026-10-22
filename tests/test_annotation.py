"""Annotation schemas: the vocabulary a dataset (one model) is labelled against.

These tests pin the *contract* a dataset depends on: which keypoints a model
system has, which pairs measure what, what the short export names are, and that a
detection-only model claims no keypoints at all. Everything else in the feature -
the canvas, the labels, the export - follows from this module, so it is tested
headless.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from daphnia import annotation


class KeypointSystemTests(unittest.TestCase):
    """Two keypoints, or four - and the interval each pair measures."""

    def test_the_two_point_system_is_head_and_tail(self):
        system = annotation.KEYPOINT_SYSTEMS["head_tail"]
        self.assertEqual(system.keypoints, (annotation.HEAD, annotation.TAIL))
        self.assertEqual(system.traits, ("length",))
        self.assertEqual(system.points_for("length"), (annotation.HEAD,
                                                       annotation.TAIL))

    def test_the_four_point_system_adds_the_sides(self):
        system = annotation.KEYPOINT_SYSTEMS["head_tail_sides"]
        self.assertEqual(
            system.keypoints,
            (annotation.HEAD, annotation.TAIL, annotation.LEFT, annotation.RIGHT))
        self.assertEqual(system.traits, ("length", "width"))
        self.assertEqual(system.points_for("width"), (annotation.LEFT,
                                                      annotation.RIGHT))

    def test_the_detection_system_has_no_keypoints(self):
        system = annotation.KEYPOINT_SYSTEMS["none"]
        self.assertEqual(system.keypoints, ())
        self.assertEqual(system.traits, ())
        self.assertEqual(system.points_for("length"), ())

    def test_a_landmark_knows_which_trait_it_belongs_to(self):
        system = annotation.KEYPOINT_SYSTEMS["head_tail_sides"]
        self.assertEqual(system.trait_of(annotation.HEAD), "length")
        self.assertEqual(system.trait_of(annotation.RIGHT), "width")
        self.assertIsNone(system.trait_of("elbow"))

    def test_the_landmark_names_are_the_study_vocabulary(self):
        """X-N08: no surface invents its own wording for the anatomy."""
        from daphnia.study import LANDMARK_TRAIT, TRAIT_LANDMARKS
        system = annotation.KEYPOINT_SYSTEMS["head_tail_sides"]
        self.assertEqual(system.points_for("length"), TRAIT_LANDMARKS["length"])
        for name in system.keypoints:
            self.assertIn(name, LANDMARK_TRAIT)


class SchemaTests(unittest.TestCase):
    """The model targets the Dataset surface offers, and what they imply."""

    def test_every_schema_key_resolves(self):
        for schema in annotation.SCHEMAS:
            self.assertIs(annotation.get(schema.key), schema)

    def test_an_unknown_key_falls_back_to_the_default(self):
        self.assertEqual(annotation.get("nope").key, annotation.DEFAULT_SCHEMA)
        self.assertEqual(annotation.get(None).key, annotation.DEFAULT_SCHEMA)

    def test_the_default_keeps_the_vocabulary_the_app_already_used(self):
        """Arriving with an existing corpus must not change what is labelled."""
        schema = annotation.default()
        self.assertEqual(schema.traits, ("length", "width"))
        self.assertTrue(schema.uses_keypoints)
        self.assertFalse(schema.exports_boxes)

    def test_the_two_point_pose_measures_length_only(self):
        schema = annotation.get("pose_head_tail")
        self.assertEqual(schema.keypoints, (annotation.HEAD, annotation.TAIL))
        self.assertEqual(schema.traits, ("length",))
        self.assertEqual(schema.label_columns(), ("keypoints",))

    def test_a_pose_with_boxes_carries_both(self):
        schema = annotation.get("pose_full_boxes")
        self.assertEqual(schema.label_columns(), ("keypoints", "bbox"))
        self.assertTrue(schema.exports_boxes)
        self.assertFalse(schema.is_detection_only)

    def test_detection_only_labels_nothing_but_a_box(self):
        schema = annotation.get("boxes")
        self.assertFalse(schema.uses_keypoints)
        self.assertTrue(schema.is_detection_only)
        self.assertEqual(schema.label_columns(), ("bbox",))
        self.assertEqual(schema.traits, ())

    def test_the_export_names_are_the_short_model_names(self):
        self.assertEqual(annotation.export_name(annotation.HEAD), "head")
        self.assertEqual(annotation.export_name(annotation.TAIL), "tail")
        self.assertEqual(annotation.export_name(annotation.LEFT), "sideL")
        self.assertEqual(annotation.export_name(annotation.RIGHT), "sideR")
        self.assertEqual(annotation.export_name("unknown point"), "unknown_point")

    def test_options_are_key_and_label_pairs_for_a_dropdown(self):
        options = annotation.options()
        self.assertEqual([key for key, _ in options], list(annotation.keys()))
        self.assertTrue(all(label for _key, label in options))

    def test_describe_reports_what_the_model_expects(self):
        text = annotation.get("boxes").describe()
        self.assertIn("no keypoints", text)
        self.assertIn("bounding box", text.lower())

    def test_the_box_has_a_label_and_is_a_trait_of_its_own(self):
        self.assertEqual(annotation.trait_label(annotation.BOX_TRAIT),
                         annotation.BOX_LABEL)
        self.assertIn(annotation.BOX_TRAIT, annotation.all_traits())
        self.assertEqual(annotation.all_traits()[:2], ("length", "width"))


class SkeletonTests(unittest.TestCase):
    """The skeleton: which keypoints are connected, and how that is exported.

    A pose model is not a bag of points - it is points *and* the bones between
    them. Without the bones written down, a training config has to guess the
    connectivity, and two projects that guess differently cannot compare results.
    """

    def test_the_four_point_system_has_two_bones(self):
        schema = annotation.get("pose_full")
        self.assertEqual(schema.edges, ((annotation.HEAD, annotation.TAIL),
                                        (annotation.LEFT, annotation.RIGHT)))

    def test_the_export_names_are_the_short_ones(self):
        self.assertEqual(annotation.get("pose_full").export_edges(),
                         (("head", "tail"), ("sideL", "sideR")))

    def test_a_two_point_system_has_one_bone(self):
        self.assertEqual(annotation.get("pose_head_tail").export_edges(),
                         (("head", "tail"),))

    def test_a_system_without_its_own_edges_falls_back_to_its_intervals(self):
        system = annotation.KeypointSystem(
            key="k", label="l", keypoints=("a", "b"),
            intervals=(("length", "a", "b"),), hint="h")
        self.assertEqual(system.skeleton, (("a", "b"),))

    def test_a_named_bone_that_measures_nothing_is_kept(self):
        """The reason ``edges`` exists: connectivity is not only what is measured."""
        system = annotation.KeypointSystem(
            key="k", label="l", keypoints=("a", "b", "c"),
            intervals=(("length", "a", "b"),), hint="h",
            edges=(("a", "b"), ("b", "c")))
        self.assertEqual(system.skeleton, (("a", "b"), ("b", "c")))

    def test_a_detection_only_schema_has_no_skeleton(self):
        schema = annotation.get("boxes")
        self.assertEqual(schema.edges, ())
        self.assertEqual(schema.export_edges(), ())


if __name__ == "__main__":
    unittest.main()