"""Phase 6I: live HTTP diagnostics. Standard library only; no auto approvals."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# These prompts request inspection only. This HTTP client is not a tool sandbox.
# Unexpected tool use is reported and stops later cases; enforcement remains
# in the backend capability system. No destructive goals are in these suites.
READ_TOOLS = {
    "get_system_volume", "get_frontmost_application", "get_running_applications",
    "get_finder_windows", "get_frontmost_window", "get_application_windows",
    "get_finder_window_path", "list_files", "get_finder_selection",
    "get_focused_ui_element", "list_ui_elements",
}
CASES = [
    dict(id="chat_control", suite="quick", mode="chat",
         message="What is two plus two? Answer in one sentence.", route="chat", required=[], minimum=0),
    dict(id="natural_auto", suite="quick", mode="auto",
         message="Could you tell me how loud my Mac is right now?", route="action",
         required=["get_system_volume"], minimum=1),
    dict(id="natural_forced", suite="quick", mode="agent",
         message="Could you tell me how loud my Mac is right now?", route="agent",
         required=["get_system_volume"], minimum=1),
    dict(id="two_step", suite="quick", mode="agent",
         message="Read the current system volume, then identify the frontmost application. Report both observations. Do not change anything.",
         route="agent", required=["get_system_volume", "get_frontmost_application"], minimum=2),
    dict(id="ambiguous_target", suite="full", mode="agent",
         message="Inspect the file I meant earlier. I have not provided its path in this conversation. Ask me which file; do not guess, access files, or change anything.",
         route="agent", required=[], minimum=0, expected_status="waiting_for_user", no_tools=True),
    dict(id="five_step", suite="full", mode="agent",
         message="Perform these five inspections: read system volume; identify the frontmost application; list running applications; list Finder windows; inspect the frontmost window. Report each result or the specific inspection that failed. Do not change anything.",
         route="agent", required=["get_system_volume", "get_frontmost_application", "get_running_applications", "get_finder_windows", "get_frontmost_window"], minimum=5),
    dict(id="gather_then_answer", suite="full", mode="agent",
         message="Find out which application is in front and inspect its windows. Use those observations to describe my current workspace. Do not open, close, move, resize, or change anything.",
         route="agent", required=["get_frontmost_application", "get_application_windows"], minimum=2),
]


class TransportError(RuntimeError):
    pass


class HTTP:
    def __init__(self, base_url, timeout=210):
        parsed = urllib.parse.urlparse(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                or parsed.username or parsed.password or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment):
            raise ValueError("Use a local HTTP base URL, for example http://127.0.0.1:8000")
        self.base_url, self.timeout = base_url.rstrip("/"), timeout

    def request(self, method, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.base_url + path, data=body, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            # Never retry: a timed-out POST could still be executing.
            raise TransportError(f"{type(exc).__name__}: {exc}") from exc


def answer_coverage_issues(case, response, diagnostics):
    """Check supported snapshot values, not general semantic correctness."""
    required = set(case["required"]) & {"get_system_volume", "get_frontmost_application"}
    if not required:
        return []
    issues = []
    rows = diagnostics.get("inspection_results", [])
    trace = diagnostics.get("step_trace", [])
    if not isinstance(rows, list) or not isinstance(trace, list):
        return ["Invalid inspection_results or step_trace diagnostics"]
    answer = response.get("answer") or ""
    if not isinstance(answer, str):
        answer = ""
    for capability in sorted(required):
        samples = [r for r in rows if isinstance(r, dict) and r.get("capability") == capability]
        if not samples:
            issues.append("Missing observed answer data: " + capability)
            continue
        row = samples[-1]
        if not any(isinstance(step, dict) and step.get("step_number") == row.get("step_number")
                   and step.get("capability") == capability and step.get("success") is True
                   for step in trace):
            issues.append("Answer data has no matching successful step: " + capability)
        data = row.get("data")
        if not isinstance(data, dict):
            issues.append("Invalid observed answer data: " + capability)
            continue
        if capability == "get_system_volume":
            volume, muted = data.get("volume"), data.get("muted")
            if (type(volume) not in {int, float} or not math.isfinite(volume)
                    or not 0 <= volume <= 100 or type(muted) is not bool):
                issues.append("Invalid observed volume or mute state")
                continue
            amounts = re.findall(r"(?<![\w.])([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent\b)", answer, re.I)
            if not any(float(value) == volume for value in amounts):
                issues.append("Answer omits the observed volume percentage")
            mute_pattern = (r"(?<!not )(?<!un)\bmuted\b" if muted else r"\b(?:not muted|unmuted)\b")
            if not re.search(mute_pattern, answer, re.I):
                issues.append("Answer omits the observed mute state")
        else:
            name = data.get("name")
            if not isinstance(name, str) or not name.strip():
                issues.append("Invalid observed application name")
            elif not re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", answer, re.I):
                issues.append("Answer omits the observed frontmost application")
    return issues


def evaluate(case, response):
    issues = []
    if not isinstance(response, dict):
        return {"status": "FAIL", "issues": ["Response is not a JSON object"], "stop": True}
    tools = response.get("tools_used")
    if not isinstance(tools, list) or any(not isinstance(x, str) for x in tools):
        return {"status": "FAIL", "issues": ["Invalid tools_used response field"], "stop": True}
    unexpected = sorted(set(tools) - READ_TOOLS)
    if unexpected:
        issues.append("Unexpected tools in inspection suite: " + ", ".join(unexpected))
    route = response.get("route")
    expected = case["route"]
    if route not in ({"direct", "agent"} if expected == "action" else {expected}):
        issues.append(f"Routing mismatch: expected {expected}, received {route}")
    if response.get("agent_error"):
        issues.append("Agent session error: " + str(response["agent_error"]))
    if not isinstance(response.get("answer"), str) or not response["answer"].strip():
        issues.append("No user-readable answer")
    missing = sorted(set(case["required"]) - set(tools))
    if missing:
        issues.append("Required capability not reported: " + ", ".join(missing))
    if case.get("no_tools") and tools:
        issues.append("This ambiguous goal must ask for clarification before using tools")
    agent = response.get("agent")
    if route == "agent":
        if not isinstance(agent, dict) or not isinstance(agent.get("diagnostics"), dict):
            issues.append("Missing agent metadata/diagnostics")
        else:
            diagnostics = agent["diagnostics"]
            if agent.get("status") != case.get("expected_status", "completed"):
                issues.append(f"Run did not complete: {agent.get('status')} / {agent.get('reason')}")
            if diagnostics.get("run_id") != agent.get("run_id"):
                issues.append("Run ID mismatch between envelope and diagnostics")
            successes = diagnostics.get("successful_steps", 0)
            if not isinstance(successes, int) or successes < case["minimum"]:
                issues.append(f"Insufficient successful steps: {successes}; need {case['minimum']}")
            if case["minimum"] and not (diagnostics.get("final_observation") or {}).get("success"):
                issues.append("Final observation does not report success")
            trace = diagnostics.get("step_trace", [])
            successful = {row.get("capability") for row in trace
                          if isinstance(row, dict) and row.get("success") is True} if isinstance(trace, list) else set()
            for capability in case["required"]:
                if capability not in successful:
                    issues.append("No successful traced execution: " + capability)
            issues.extend(answer_coverage_issues(case, response, diagnostics))
    elif route == "direct" and case["required"]:
        issues.append("Direct response lacks observation diagnostics; answer cannot be verified")
    return {"status": "FAIL" if issues else "PASS", "issues": issues,
            "stop": bool(unexpected),
            "manual_review": "Check that the answer accurately reports observed data and satisfies every requested part. Automated PASS is not semantic proof."}


def run_case(client, case):
    record = {"case": case, "started_utc": datetime.now(timezone.utc).isoformat()}
    started = time.perf_counter()
    try:
        conversation = client.request("POST", "/conversations", {})
        conversation_id = conversation["id"]
        record["conversation_id"] = conversation_id
        payload = {"message": case["message"], "mode": case["mode"], "conversation_id": conversation_id}
        record["request"] = payload
        response = client.request("POST", "/chat", payload)
        record["response"] = response
        record["evaluation"] = evaluate(case, response)
        agent = response.get("agent") if isinstance(response, dict) else None
        if isinstance(agent, dict) and agent.get("status") in {"waiting_for_confirmation", "waiting_for_user"}:
            # End a paused diagnostic run without approving its proposed action.
            cancellation = {"message": "cancel", "conversation_id": conversation_id,
                            "run_id": agent.get("run_id"),
                            "confirmation_id": agent.get("confirmation_id"),
                            "resume_token": agent.get("resume_token")}
            record["cancellation_request"] = cancellation
            cancelled = client.request("POST", "/chat", cancellation)
            record["cancellation_response"] = cancelled
            if (cancelled.get("agent") or {}).get("status") != "cancelled":
                record["evaluation"]["issues"].append("Diagnostic cancellation was not acknowledged")
                record["evaluation"]["status"] = "FAIL"
                record["evaluation"]["stop"] = True
        # Legacy direct confirmation has no ID metadata: do not risk an
        # unbound cleanup message. Stop for user inspection if action absent.
        if response.get("route") == "direct" and not response.get("tools_used"):
            record["evaluation"]["issues"].append("Direct response reported no tool; inspect it before continuing")
            record["evaluation"]["status"] = "FAIL"
            record["evaluation"]["stop"] = True
    except (TransportError, KeyError, TypeError, AttributeError) as exc:
        record["evaluation"] = {"status": "ERROR", "issues": [str(exc)], "stop": True,
            "manual_review": "No automatic retry. Check backend logs: a timed-out request may still be running."}
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("quick", "full"), default="quick")
    parser.add_argument("--case", choices=[c["id"] for c in CASES])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Run existing agent unit/integration suites instead of HTTP")
    args = parser.parse_args(argv)
    if args.list:
        for case in CASES:
            print(case["id"], "[" + case["mode"] + "]", case["message"])
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path("diagnostics") / ("agent_6i_" + stamp + ".json")
    report = {"phase": "6I", "diagnostic_revision": "6I.1", "created_utc": stamp, "python": sys.version,
              "platform": platform.platform(), "mode": "offline" if args.offline else "live_http",
              "milestone": "NOT_SIGNED_OFF", "cases": []}
    if args.offline:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "core/tests",
                                 "-p", "test_agent*.py", "-v"], cwd=root, text=True, capture_output=True)
        match = re.search(r"Ran (\d+) tests?", result.stderr)
        count = int(match.group(1)) if match else 0
        report["offline"] = {"returncode": result.returncode, "tests_run": count,
                             "minimum_expected": 187, "stdout": result.stdout, "stderr": result.stderr}
        if count < 187:
            report["offline"]["issue"] = "Fewer than 187 baseline tests discovered. Check both test_agent_termination.py and test_agent_integration.py."
        save_report(output, report)
        print(result.stderr)
        print("Report:", output.resolve())
        return result.returncode or (1 if count < 187 else 0)
    selected = [c for c in CASES if (c["id"] == args.case if args.case else args.suite == "full" or c["suite"] == "quick")]
    report["planned_cases"] = [c["id"] for c in selected]
    client = HTTP(args.base_url)
    try:
        report["health"] = client.request("GET", "/health")
        if not isinstance(report["health"], dict) or report["health"].get("status") != "ok":
            raise TransportError("Backend health check did not report ok")
        for case in selected:
            print("Running", case["id"], "...", flush=True)
            record = run_case(client, case)
            report["cases"].append(record)
            save_report(output, report)
            evaluation = record["evaluation"]
            print(evaluation["status"], "—", record["elapsed_seconds"], "seconds")
            for issue in evaluation["issues"]:
                print("  " + issue)
            if evaluation.get("stop"):
                report["stopped_early"] = True
                break
    except (TransportError, KeyboardInterrupt) as exc:
        report["interrupted"] = type(exc).__name__ + ": " + str(exc)
    finally:
        outcomes = {row["case"]["id"]: row["evaluation"]["status"] for row in report["cases"]}
        if outcomes.get("natural_auto") == "FAIL" and outcomes.get("natural_forced") == "PASS":
            report["interpretation"] = "Forced agent passed while automatic wording failed. Inspect the recorded route: automatic routing is the first issue to investigate."
        elif outcomes.get("natural_forced") in {"FAIL", "ERROR"}:
            report["interpretation"] = "Forced agent did not pass. Inspect termination reason, failure categories, tools and backend planner logs before changing only the router."
        else:
            report["interpretation"] = "Review the case checks and compare each answer with its observation. Milestone sign-off is still manual."
        save_report(output, report)
        print("Report:", output.resolve())
    return 0 if len(report["cases"]) == len(selected) and all(r["evaluation"]["status"] == "PASS" for r in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
