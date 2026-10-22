"""Study templates: versioned, reusable starting points and the protocol on a study."""

import tempfile
import unittest
from pathlib import Path

from daphnia import protocols
from daphnia.db import connection, init_db, queries, repository
from daphnia.templates import StudyTemplate


class TemplateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "test.db"
        self.destination = Path(self._tmp.name) / "images"
        init_db(self.db_path)

    def test_saving_under_a_name_appends_a_version(self):
        with connection(self.db_path) as conn:
            first = repository.save_study_template(conn, name="Daily", target_mode="plates")
            second = repository.save_study_template(conn, name="Daily", target_mode="plates")
            rows = queries.list_study_templates(conn)
        self.assertNotEqual(first, second)
        self.assertEqual(len(rows), 1)                 # only the latest version is listed
        self.assertEqual(rows[0]["version"], 2)

    def test_a_template_round_trips_through_the_database(self):
        with connection(self.db_path) as conn:
            template_id = repository.save_study_template(
                conn, name="Plant", organism="Lemna", organism_kind="plant",
                target_mode="plates", timecourse="single",
                measurement_interval_hours=0.0, protocol_key="careful")
            template = StudyTemplate.from_row(
                repository.template_by_id(conn, template_id))
        self.assertEqual((template.organism, template.organism_kind), ("Lemna", "plant"))
        self.assertEqual(template.target_mode, "plates")
        self.assertEqual(template.protocol.key, "careful")
        self.assertEqual(template.label(), "Plant v1")

    def test_two_names_are_listed_independently(self):
        with connection(self.db_path) as conn:
            repository.save_study_template(conn, name="A")
            repository.save_study_template(conn, name="B")
            rows = queries.list_study_templates(conn)
        self.assertEqual(sorted(row["name"] for row in rows), ["A", "B"])

    def test_a_study_remembers_the_protocol_it_was_created_with(self):
        with connection(self.db_path) as conn:
            repository.create_study(conn, name="Trial", destination=self.destination,
                                    protocol_key="careful")
            spec = queries.active_study_spec(conn)
        self.assertEqual(spec.protocol_key, "careful")
        self.assertEqual(spec.protocol.burst_size,
                         protocols.get("careful").burst_size)

    def test_a_study_defaults_to_the_default_protocol(self):
        with connection(self.db_path) as conn:
            repository.create_study(conn, name="Trial", destination=self.destination)
            spec = queries.active_study_spec(conn)
        self.assertEqual(spec.protocol_key, protocols.DEFAULT_PROTOCOL)


if __name__ == "__main__":
    unittest.main()
