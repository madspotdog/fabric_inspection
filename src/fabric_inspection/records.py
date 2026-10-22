# -*- coding: utf-8 -*-
"""检测台账：CSV 追加、编号续接、图片归档与失败回滚。"""
from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

CSV_HEADER = [
    "检测时间", "样品编号", "棉含量档位", "纯面料结果", "交叉结论",
    "棉含量置信度", "纯面料置信度", "棉门控概率", "清晰度", "存样图",
]


class RecordWriter:
    """一个目录只应由一个进程写入。"""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.report = self.folder / "inspection_report.csv"
        self._sample_id = self._read_last_id() + 1

    def _read_last_id(self) -> int:
        if not self.report.exists() or self.report.stat().st_size == 0:
            return 0
        last = 0
        with self.report.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != CSV_HEADER:
                raise RuntimeError("台账列结构不兼容，请使用新的归档目录")
            for row in reader:
                value = row["样品编号"].rsplit("-", 1)[-1]
                last = max(last, int(value))
        return last

    def allocate(self) -> int:
        sample_id = self._sample_id
        self._sample_id += 1
        return sample_id

    def save(self, result, image_bytes: bytes | None = None,
             sample_id: int | None = None) -> dict:
        """写入一行记录；提供 JPEG 字节时同时保存样图。

        返回写入的字段字典，便于调用方显示或审计。
        """
        sample_id = sample_id or self.allocate()
        timestamp = datetime.now()
        image_name = ""
        image_path = None
        if image_bytes:
            image_name = (f"Sample{sample_id:03d}_"
                          f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg")
            image_path = self.folder / image_name
            image_path.write_bytes(image_bytes)

        row = {
            "检测时间": timestamp.isoformat(timespec="seconds"),
            "样品编号": f"Sample-{sample_id:03d}",
            "棉含量档位": result.cotton_tier,
            "纯面料结果": result.pure_fiber,
            "交叉结论": result.cross_check,
            "棉含量置信度": f"{result.cotton_confidence:.2%}",
            "纯面料置信度": f"{result.fiber_confidence:.2%}",
            "棉门控概率": f"{result.cotton_gate_probability:.2%}",
            "清晰度": f"{result.focus:.1f}",
            "存样图": image_name,
        }

        previous_size = self.report.stat().st_size if self.report.exists() else 0
        need_header = previous_size == 0
        try:
            with self.report.open("a", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_HEADER)
                if need_header:
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            if self.report.exists():
                with self.report.open("r+b") as stream:
                    stream.truncate(previous_size)
            if image_path and image_path.exists():
                image_path.unlink()
            raise
        return {"sample_id": sample_id, "image": image_name, "row": row}
