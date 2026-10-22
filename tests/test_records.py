# -*- coding: utf-8 -*-
import csv

import pytest

from fabric_inspection.decision import decide, Thresholds
from fabric_inspection.records import CSV_HEADER, RecordWriter

FIBERS = ["丝", "亚麻", "其他", "棉", "涤纶", "羊毛"]


def make_result():
    fiber = [0.02, 0.02, 0.06, 0.80, 0.05, 0.05]
    cotton = [0.05, 0.05, 0.90]
    return decide(cotton, fiber, FIBERS, Thresholds())


def test_header_and_first_row(tmp_path):
    writer = RecordWriter(tmp_path / "records")
    saved = writer.save(make_result(), image_bytes=b"jpeg")
    assert saved["sample_id"] == 1
    assert saved["image"].startswith("Sample001_")
    with (tmp_path / "records" / "inspection_report.csv").open(
            newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["样品编号"] == "Sample-001"
    assert rows[0]["棉含量档位"] == "高棉"
    assert rows[0]["交叉结论"] == "通过"


def test_numbering_continues(tmp_path):
    folder = tmp_path / "records"
    RecordWriter(folder).save(make_result())
    writer = RecordWriter(folder)
    saved = writer.save(make_result())
    assert saved["sample_id"] == 2


def test_incompatible_header_rejected(tmp_path):
    folder = tmp_path / "records"
    folder.mkdir()
    (folder / "inspection_report.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        RecordWriter(folder)


def test_failed_write_rolls_back(tmp_path, monkeypatch):
    folder = tmp_path / "records"
    writer = RecordWriter(folder)
    original = writer.report.open

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(writer.report.__class__, "open", broken, raising=False)
    with pytest.raises(OSError):
        writer.save(make_result(), image_bytes=b"jpeg")
    assert not list(folder.glob("*.jpg"))
