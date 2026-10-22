"""Study templates: reusable, versioned starting points for a study.

A template captures the decisions that repeat from one run to the next - what is
measured, how often, and with which acquisition protocol - so a longitudinal
experiment can be restarted in seconds instead of retyped. Templates are
*versioned by name*: saving under an existing name appends a new version and the
older ones stay as history, so a template is never overwritten in place.
"""

from dataclasses import dataclass

from daphnia import protocols


@dataclass(frozen=True)
class StudyTemplate:
    """A row of ``study_templates``, as the Setup screen reads it."""

    template_id: str
    name: str
    version: int
    organism: str
    organism_kind: str
    target_mode: str
    timecourse: str
    measurement_interval_hours: float
    protocol_key: str
    created_at: str = ""

    @classmethod
    def from_row(cls, row) -> "StudyTemplate":
        """Build a template from a ``study_templates`` row."""
        return cls(
            template_id=row["template_id"],
            name=row["name"],
            version=row["version"],
            organism=row["organism"],
            organism_kind=row["organism_kind"],
            target_mode=row["target_mode"],
            timecourse=row["timecourse"],
            measurement_interval_hours=row["measurement_interval_hours"],
            protocol_key=row["protocol_key"],
            created_at=row["created_at"],
        )

    @property
    def protocol(self) -> protocols.Protocol:
        """The acquisition protocol this template was saved with."""
        return protocols.get(self.protocol_key)

    def label(self) -> str:
        """Combo-box label, e.g. ``'Daily Daphnia v2'``."""
        return f"{self.name} v{self.version}"
