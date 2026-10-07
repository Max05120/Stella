"""Tests for the diagnostic runner itself; no backend required."""
import copy
import unittest

from agent_diagnostics import CASES, HTTP, TransportError, evaluate, run_case


def response(status="completed", tools=None, successes=1):
    tools = ["get_system_volume"] if tools is None else tools
    rows = []
    for number, capability in enumerate(tools, 1):
        if capability == "get_system_volume":
            rows.append({"step_number": number, "capability": capability,
                         "data": {"volume": 20, "muted": False}})
        elif capability == "get_frontmost_application":
            rows.append({"step_number": number, "capability": capability, "data": {"name": "Code"}})
    return {"answer": "Volume is 20%, not muted. Frontmost application: Code.", "route": "agent", "tools_used": tools,
            "agent": {"run_id": "run-1", "status": status, "reason": None,
                      "confirmation_id": "confirm-1" if status == "waiting_for_confirmation" else None,
                      "resume_token": "question-1" if status == "waiting_for_user" else None,
                      "diagnostics": {"run_id": "run-1", "successful_steps": successes,
                                      "final_observation": {"success": True, "data": 20},
                                      "inspection_results": rows,
                                      "step_trace": [{"step_number": n, "capability": c, "success": True}
                                                     for n, c in enumerate(tools, 1)]}}}


class FakeHTTP:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []
    def request(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        item = self.results.pop(0)
        if isinstance(item, Exception): raise item
        return item


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.case = next(c for c in CASES if c["id"] == "natural_forced")

    def test_completed_without_tools_cannot_pass(self):
        result = response()
        result["tools_used"] = []
        result["agent"]["diagnostics"]["successful_steps"] = 0
        self.assertEqual(evaluate(self.case, result)["status"], "FAIL")

    def test_correct_observation_passes_with_manual_review_required(self):
        result = evaluate(self.case, response())
        self.assertEqual(result["status"], "PASS")
        self.assertIn("not semantic proof", result["manual_review"])

    def test_auto_route_gap_is_recorded(self):
        case = next(c for c in CASES if c["id"] == "natural_auto")
        result = evaluate(case, {"answer": "Maybe 50", "route": "chat", "tools_used": []})
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("Routing mismatch" in x for x in result["issues"]))

    def test_unexpected_mutation_stops_later_cases(self):
        result = evaluate(self.case, response(tools=["set_system_volume"]))
        self.assertTrue(result["stop"])
        self.assertEqual(result["status"], "FAIL")

    def test_pending_confirmation_is_cancelled_never_approved(self):
        fake = FakeHTTP({"id": "chat-1"}, response("waiting_for_confirmation"), response("cancelled"))
        record = run_case(fake, self.case)
        self.assertEqual(fake.calls[-1][2]["message"], "cancel")
        self.assertEqual(fake.calls[-1][2]["confirmation_id"], "confirm-1")
        self.assertEqual(fake.calls[-1][2]["run_id"], "run-1")
        self.assertEqual(record["response"]["agent"]["status"], "waiting_for_confirmation")

    def test_transport_failure_is_not_retried(self):
        fake = FakeHTTP({"id": "chat-1"}, TransportError("timeout"))
        record = run_case(fake, self.case)
        self.assertEqual(len(fake.calls), 2)
        self.assertTrue(record["evaluation"]["stop"])
        self.assertEqual(record["evaluation"]["status"], "ERROR")

    def test_cancellation_failure_stops_suite(self):
        fake = FakeHTTP({"id": "chat-1"}, response("waiting_for_confirmation"), response("waiting_for_confirmation"))
        record = run_case(fake, self.case)
        self.assertTrue(record["evaluation"]["stop"])

    def test_five_step_requires_all_tools_and_success_count(self):
        case = next(c for c in CASES if c["id"] == "five_step")
        result = response(tools=list(case["required"]), successes=1)
        self.assertEqual(evaluate(case, result)["status"], "FAIL")
        result["agent"]["diagnostics"]["successful_steps"] = 5
        self.assertEqual(evaluate(case, result)["status"], "PASS")

    def test_ambiguity_requires_question_without_tool_call(self):
        case = next(c for c in CASES if c["id"] == "ambiguous_target")
        result = response("waiting_for_user", successes=0)
        result["tools_used"] = []
        self.assertEqual(evaluate(case, result)["status"], "PASS")
        result["tools_used"] = ["list_files"]
        self.assertEqual(evaluate(case, result)["status"], "FAIL")

    def test_mismatched_run_metadata_fails(self):
        result = response()
        result["agent"]["diagnostics"]["run_id"] = "other"
        self.assertEqual(evaluate(self.case, result)["status"], "FAIL")

    def test_remote_server_is_not_accepted(self):
        with self.assertRaises(ValueError):
            HTTP("https://example.com")
        HTTP("http://127.0.0.1:8000")

    def test_generic_two_step_completion_is_a_failure(self):
        case = next(c for c in CASES if c["id"] == "two_step")
        result = response(tools=case["required"], successes=2)
        result["answer"] = "The requested task is complete."
        check = evaluate(case, result)
        self.assertEqual(check["status"], "FAIL")
        self.assertIn("Answer omits the observed volume percentage", check["issues"])
        self.assertIn("Answer omits the observed frontmost application", check["issues"])

    def test_two_step_with_both_values_passes(self):
        case = next(c for c in CASES if c["id"] == "two_step")
        self.assertEqual(evaluate(case, response(tools=case["required"], successes=2))["status"], "PASS")

    def test_wrong_volume_and_mute_state_fail(self):
        result = response()
        result["answer"] = "Volume is 87%, muted."
        check = evaluate(self.case, result)
        self.assertEqual(check["status"], "FAIL")
        self.assertIn("Answer omits the observed volume percentage", check["issues"])
        self.assertIn("Answer omits the observed mute state", check["issues"])

    def test_legacy_diagnostics_without_evidence_cannot_pass(self):
        result = response()
        result["agent"]["diagnostics"].pop("inspection_results")
        result["agent"]["diagnostics"].pop("step_trace")
        self.assertEqual(evaluate(self.case, result)["status"], "FAIL")

    def test_failed_tool_cannot_supply_verified_answer(self):
        result = response()
        result["agent"]["diagnostics"]["step_trace"][0]["success"] = False
        self.assertEqual(evaluate(self.case, result)["status"], "FAIL")

    def test_percentage_substring_is_not_a_match(self):
        result = response()
        result["answer"] = "Volume is 120%, not muted."
        self.assertEqual(evaluate(self.case, result)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
