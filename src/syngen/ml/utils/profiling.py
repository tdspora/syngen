from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil


def _now_iso() -> str:
    # ISO-ish timestamp without timezone (consistent with existing logging style)
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


@dataclass
class ProfileLogger:
    """Lightweight profiling logger for wall time + process memory.

    Records a time series of events with RSS/VMS so we can spot memory spikes
    during training and postprocessing.
    """

    enabled: bool
    out_path: Path
    process_id: int = field(default_factory=os.getpid)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter)

    def log(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return

        proc = psutil.Process(self.process_id)
        mem = proc.memory_info()
        self.rows.append(
            {
                "ts": _now_iso(),
                "t_rel_s": round(time.perf_counter() - self._t0, 6),
                "event": event,
                "rss_bytes": int(getattr(mem, "rss", 0)),
                "vms_bytes": int(getattr(mem, "vms", 0)),
                **fields,
            }
        )

    def flush(self) -> Optional[Path]:
        if not self.enabled:
            return None

        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        # Avoid importing pandas in core training hot path.
        if not self.rows:
            self.out_path.write_text("ts,t_rel_s,event,rss_bytes,vms_bytes\n")
            return self.out_path

        # Determine a stable column order.
        columns: List[str] = []
        seen = set()
        for row in self.rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    columns.append(key)

        def _csv_escape(val: Any) -> str:
            s = "" if val is None else str(val)
            if any(c in s for c in [",", "\n", "\r", '"']):
                s = '"' + s.replace('"', '""') + '"'
            return s

        lines = [",".join(columns)]
        for row in self.rows:
            lines.append(",".join(_csv_escape(row.get(c)) for c in columns))

        self.out_path.write_text("\n".join(lines) + "\n")
        return self.out_path
