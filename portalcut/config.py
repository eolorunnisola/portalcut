"""Per-job processing settings, stored in the job's meta["config"]."""

from dataclasses import asdict, dataclass, fields

TRACKERS = ("bytetrack", "botsort", "ocsort")


@dataclass
class Config:
    proxy_height: int = 720          # processing resolution (the proxy video's height)
    detect_fps: float = 10.0         # 10 fps cut identity swaps for every tracker tested (docs/ARCHITECTURE.md)
    person_conf: float = 0.4         # drop person detections below this confidence
    ball_conf: float = 0.25          # COCO's ball class is weak on basketball; keep low
    tracker: str = "botsort"         # fewest identity swaps measured; "bytetrack" keeps ~20% more detections
    lost_seconds: float = 1.0        # longer buffers re-attach lost IDs to the wrong player; linking repairs gaps
    tracker_params: str = ""         # extra tracker settings, "name=value,..." (see trackers.BoTSORTTracker)
    annotate_fps: float = 10.0       # annotated video frame rate (boxes interpolated between detections)
    jersey_numbers: bool = True      # read numbers with Claude Haiku (~$0.03/game, capped at $0.25) to link tracklets
    gemini: bool = False             # optional Gemini Flash analysis after tracking
    gemini_interval_s: float = 30.0  # one sampled frame every this many seconds

    def __post_init__(self):
        if not 240 <= self.proxy_height <= 2160:
            raise ValueError("proxy_height must be between 240 and 2160")
        if not 1 <= self.detect_fps <= 30:
            raise ValueError("detect_fps must be between 1 and 30")
        for name in ("person_conf", "ball_conf"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.tracker not in TRACKERS:
            raise ValueError(f"tracker must be one of {TRACKERS}")
        if not 0 < self.lost_seconds <= 30 or not 1 <= self.annotate_fps <= 30 or self.gemini_interval_s < 5:
            raise ValueError("lost_seconds 0-30, annotate_fps 1-30, gemini_interval_s >= 5")
        parse_params(self.tracker_params)  # raises on a malformed string

    @classmethod
    def from_dict(cls, d: dict | None) -> "Config":
        """Unknown keys are ignored; values are converted to the field's type (query strings arrive as text)."""
        d = d or {}
        kw = {}
        for f in fields(cls):
            if f.name in d and d[f.name] is not None:
                v = d[f.name]
                kw[f.name] = (str(v).lower() in ("1", "true", "yes", "on")) if f.type in (bool, "bool") else \
                    (int(v) if f.type in (int, "int") else float(v) if f.type in (float, "float") else str(v))
        return cls(**kw)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_params(text: str) -> dict:
    """"a=0.4,b=false,c=orb" -> {"a": 0.4, "b": False, "c": "orb"}."""
    out = {}
    for kv in filter(None, (x.strip() for x in text.split(","))):
        k, sep, v = kv.partition("=")
        if not sep or not k.isidentifier():
            raise ValueError(f"tracker_params: expected name=value, got {kv!r}")
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = {"true": True, "false": False}.get(v.lower(), v)
    return out
