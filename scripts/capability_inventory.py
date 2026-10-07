"""Read registered tool metadata for Phase 7. Never invoke a capability handler."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def build_inventory(adapter):
    rows = [asdict(tool) for tool in adapter.list_tools()]
    rows.sort(key=lambda row: (row["family"], row["name"]))
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "registered_metadata_only",
        "note": "Registration does not prove permission, availability, timeout behavior or successful execution.",
        "count": len(rows),
        "capabilities": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from core.agent.tool_adapter import AgentToolAdapter
    report = build_inventory(AgentToolAdapter())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or root / "diagnostics" / ("capability_inventory_" + stamp + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    print(f"Registered capabilities: {report['count']}")
    print("Inventory:", output.resolve())


if __name__ == "__main__":
    main()