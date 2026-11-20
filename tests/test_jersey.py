"""Jersey reader with Claude replaced by a fake: these tests never call the API or spend money."""
import json
from types import SimpleNamespace

from PIL import Image

from portalcut import jersey
from portalcut.events import HAIKU


class FakeClient:
    def __init__(self, stop="end_turn", number="13"):
        self.calls = []
        self.stop, self.number = stop, number
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kw):
        self.calls.append(kw)
        tids = [int(b["text"][6:-1]) for b in kw["messages"][0]["content"] if b["type"] == "text"]
        parsed = jersey.Batch(tracks=[jersey.TrackReading(track_id=t, readings=[self.number, self.number, ""])
                                      for t in tids])
        return SimpleNamespace(usage=SimpleNamespace(input_tokens=3000, output_tokens=100), stop_reason=self.stop,
                               parsed_output=parsed if self.stop != "refusal" else None)


def _strips(tmp_path, tids):
    (tmp_path / "crops").mkdir(exist_ok=True)
    for t in tids:
        Image.new("RGB", (128 * 2 + 6, 128), (t * 7 % 256, 90, 40)).save(tmp_path / "crops" / f"{t}_torso.jpg")


def test_vote():
    assert jersey.vote(["13", "13", ""]) == {"number": "13", "votes": 2, "conflict": False}
    assert jersey.vote(["13"])["number"] == "13"
    assert jersey.vote(["13", "18"]) == {"number": None, "votes": 1, "conflict": True}
    assert jersey.vote(["", ""]) == {"number": None, "votes": 0, "conflict": False}
    assert jersey.vote(["13", "13", "18"]) == {"number": "13", "votes": 2, "conflict": True}


def test_read_batch_request_and_cache(tmp_path):
    _strips(tmp_path, [4, 9])
    c = FakeClient()
    r = jersey.read_batch(tmp_path, [4, 9], c)
    kw = c.calls[0]
    content = kw["messages"][0]["content"]
    assert [b["text"] for b in content if b["type"] == "text"] == ["Track 4:", "Track 9:"]
    assert sum(b["type"] == "image" for b in content) == 2
    assert kw["model"] == HAIKU and kw["output_format"] is jersey.Batch
    assert "temperature" not in kw and "tool_choice" not in kw
    assert r["readings"] == {4: ["13", "13", ""], 9: ["13", "13", ""]} and r["cost"] > 0 and not r["cached"]
    r2 = jersey.read_batch(tmp_path, [4, 9], c)
    assert r2["cost"] == 0 and r2["cached"] and r2["readings"] == r["readings"] and len(c.calls) == 1


def test_read_numbers_batches_skips_and_writes(tmp_path):
    tids = list(range(25))
    _strips(tmp_path, tids)
    c = FakeClient()
    out = jersey.read_numbers(tmp_path, tids + [99], budget=10, client=c)  # 99 has no strip: skipped
    assert len(c.calls) == 3 and out["read"] == 25 and out["api_calls"] == 3 and out["skipped_over_budget"] == 0
    saved = json.loads((tmp_path / "numbers.json").read_text())
    assert len(saved) == 25 and saved["7"]["number"] == "13" and saved["7"]["readings"] == ["13", "13", ""]
    assert jersey.load_numbers(tmp_path)[7]["votes"] == 2
    again = jersey.read_numbers(tmp_path, tids, budget=10, client=c)
    assert len(c.calls) == 3 and again["read"] == 0 and again["api_calls"] == 0


def test_read_numbers_budget(tmp_path):
    _strips(tmp_path, range(25))
    c = FakeClient()
    out = jersey.read_numbers(tmp_path, list(range(25)), budget=1e-9, client=c)
    assert len(c.calls) == 1 and out["read"] == 10 and out["skipped_over_budget"] == 15


def test_refusal_gives_no_number(tmp_path):
    _strips(tmp_path, [3])
    out = jersey.read_numbers(tmp_path, [3], client=FakeClient(stop="refusal"))
    assert out["read"] == 1
    assert jersey.load_numbers(tmp_path)[3] == {"number": None, "votes": 0, "conflict": False, "readings": []}


def test_load_numbers_missing(tmp_path):
    assert jersey.load_numbers(tmp_path) == {}
