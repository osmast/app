"""Acquisition protocols: named capture + QA bundles with an explicit trade-off.

A protocol answers a single question for the operator - *how much effort and disk
do I spend per well?* - and encodes the consequence, so the choice is informed
instead of hidden in scattered settings. A study stores the protocol it was
acquired with (``studies.protocol_key``), so a later reader knows how the frames
were taken, exactly the way the optical profile is frozen on the session.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Protocol:
    """One named acquisition recipe."""

    key: str
    label: str
    burst_size: int          # frames captured per burst
    min_sharpness: float     # sharpness below this is a weak frame (QA hint)
    tradeoff: str            # one sentence, shown in the UI


#: The protocols offered everywhere, keyed by the value stored in the database.
PROTOCOLS = {
    "quick": Protocol(
        key="quick",
        label="Quick scan",
        burst_size=3,
        min_sharpness=0.0,
        tradeoff="Fewest frames per well - fastest and least disk, "
                 "but a blurred well may yield no usable frame.",
    ),
    "standard": Protocol(
        key="standard",
        label="Standard",
        burst_size=5,
        min_sharpness=5.0,
        tradeoff="Balanced: five frames per well, the sharpest one is kept.",
    ),
    "careful": Protocol(
        key="careful",
        label="Careful",
        burst_size=8,
        min_sharpness=10.0,
        tradeoff="Most frames per well - slower and uses more disk, "
                 "best odds of a sharp frame for a difficult specimen.",
    ),
}

#: Protocol used when a study does not name one.
DEFAULT_PROTOCOL = "standard"


def get(key: str) -> Protocol:
    """The protocol for ``key``, falling back to the default one."""
    return PROTOCOLS.get(key, PROTOCOLS[DEFAULT_PROTOCOL])


def labels() -> list:
    """Protocol labels, in declaration order (for a UI combo box)."""
    return [protocol.label for protocol in PROTOCOLS.values()]


def key_for_label(label: str) -> str:
    """Protocol key for a label, falling back to the default key."""
    for key, protocol in PROTOCOLS.items():
        if protocol.label == label:
            return key
    return DEFAULT_PROTOCOL
