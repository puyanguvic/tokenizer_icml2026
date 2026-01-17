import json

from ctok_extras.datasets.config import DatasetConfig
from ctok_extras.datasets.io import load_dataset


def test_load_jsonl(tmp_path):
    records = [
        {"text": "GET /", "label": "benign"},
        {"text": "POST /admin", "label": "attack"},
    ]
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    config = DatasetConfig(name="test", path=str(path), format="jsonl")
    examples = load_dataset(config)
    assert len(examples) == 2
    assert examples[0].text == "GET /"
    assert examples[0].label == "benign"


def test_load_tsv(tmp_path):
    path = tmp_path / "data.tsv"
    path.write_text("label\ttext\nbenign\tGET /\nattack\tPOST /admin\n", encoding="utf-8")

    config = DatasetConfig(
        name="test",
        path=str(path),
        format="tsv",
        has_header=True,
        columns=["label", "text"],
    )
    examples = load_dataset(config)
    assert len(examples) == 2
    assert examples[1].text == "POST /admin"
    assert examples[1].label == "attack"
