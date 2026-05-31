from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MusicEvent:
    """One generated event in the emotion-conditioned music MDP."""

    tempo: int
    position: int
    pitch: int
    duration: float
    velocity: int

    def to_dict(self):
        return asdict(self)
