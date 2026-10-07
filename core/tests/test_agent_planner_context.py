"""6I.2 regression checks with injected Ollama and tool outcomes."""
import contextlib
from copy import deepcopy
import io
import json
import types
import unittest
from unittest.mock import patch
try:
    from .test_agent_termination import M, L, P, MetadataTools, decision, observation
except ImportError:
    from test_agent_termination import M, L, P, MetadataTools, decision, observation


class WindowTools(MetadataTools):
    def list_tools(self):
        result = super().list_tools()
        for name in ("get_frontmost_application", "get_application_windows", "get_running_applications", "get_finder_windows", "get_frontmost_window"):
            parameters = [types.SimpleNamespace(name="application", type_name="str", required=True, default=None)] if name == "get_application_windows" else []
            result.append(types.SimpleNamespace(name=name, family="windows", description="Inspect macOS", risk="low", requires_confirmation=False, parameters=parameters))
        return result
    def validate_arguments(self, capability, arguments):
        if capability == "get_application_windows" and not isinstance(arguments.get("application"), str):
            return False, "application is required"
        return True, None


class PlannerContextTests(unittest.TestCase):
    def setUp(self):
        self.tools = WindowTools()
        self.planner = P.AgentPlanner(self.tools)
        self.state = M.AgentState(M.AgentGoal("Inspect the frontmost application and its windows"))
    def add(self, capability, data, success=True):
        self.state.steps.append(M.AgentStep(len(self.state.steps) + 1, decision(capability=capability), observation(success, data=data)))
    def test_large_inventory_is_bounded_and_original_preserved(self):
        data = {"count": 180, "applications": [{"name": "App" + str(i), "path": "x" * 500} for i in range(180)]}
        self.add("get_running_applications", data)
        before = deepcopy(self.state.steps)
        prompt = self.planner._build_prompt(self.state, self.tools.list_tools())
        self.assertLess(len(prompt), 18000)
        self.assertIn('"count": 180', prompt)
        self.assertIn("_stella_preview", prompt)
        self.assertNotIn("x" * 500, prompt)
        self.assertEqual(self.state.steps, before)
    def test_empty_list_is_complete_evidence(self):
        self.assertEqual(json.loads(P.preview_json({"windows": [], "count": 0})), {"count": 0, "windows": []})
    def test_long_identifier_is_marked_and_json_remains_valid(self):
        preview = json.loads(P.preview_json({"path": "/" + "x" * 9000}, 400))
        self.assertTrue(preview["path"]["_stella_preview"])
        self.assertEqual(preview["path"]["original_chars"], 9001)
    def test_frontmost_target_and_required_argument_reach_planner(self):
        self.add("get_frontmost_application", {"name": "Safari", "bundle_identifier": "com.apple.Safari"})
        prompt = self.planner._build_prompt(self.state, self.tools.list_tools())
        self.assertIn('Observed frontmost application name: "Safari"', prompt)
        self.assertIn("get_application_windows(application:str*)", prompt)
        self.assertIn("get_finder_windows describes Finder only", prompt)
    def test_mutation_invalidates_target_hint(self):
        self.add("get_frontmost_application", {"name": "Safari"})
        self.add("open_application", {"application": "Finder"})
        self.assertNotIn("Observed frontmost application name", self.planner._inspection_ledger(self.state))
    def test_failed_read_does_not_bind_target(self):
        self.add("get_frontmost_application", {"name": "Safari"}, False)
        self.assertNotIn("Observed frontmost application name", self.planner._inspection_ledger(self.state))
    def test_repair_uses_bounded_history_and_original_goal(self):
        self.add("get_running_applications", {"applications": ["large" * 1000] * 100, "count": 100})
        prompt = self.planner._build_repair_prompt(self.state, self.tools.list_tools(), "bad" * 10000, "Invalid decision")
        self.assertLess(len(prompt), 18000)
        self.assertIn(self.state.goal.text, prompt)
        self.assertIn("'correct' and 'repair' are not decision types", prompt)
    def test_schema_transport_and_metrics_without_private_content(self):
        output = io.StringIO()
        reply = {"message": {"content": '{"decision_type":"ask_user","message":"Which app?"}'}, "prompt_eval_count": 321, "eval_count": 20, "done_reason": "stop"}
        with patch.object(P.ollama, "chat", return_value=reply) as chat, contextlib.redirect_stdout(output):
            self.planner._call_model("PRIVATE GOAL DATA")
        kwargs = chat.call_args.kwargs
        schema = kwargs["format"]
        self.assertIsInstance(schema, dict)
        self.assertNotIn(
            "correct",
            schema["anyOf"][1]["properties"]["decision_type"]["enum"],
        )
        self.assertIn(
            "get_application_windows",
            schema["anyOf"][0]["properties"]["capability"]["enum"],
        )
        self.assertEqual(kwargs["options"]["num_ctx"], 8192)
        self.assertIn('"prompt_eval_count": 321', output.getvalue())
        self.assertNotIn("PRIVATE GOAL DATA", output.getvalue())
    def test_invalid_decision_rejected_even_if_transport_ignores_schema(self):
        with self.assertRaises(P.AgentPlannerError):
            self.planner._parse_decision('{"decision_type":"correct"}', self.tools.list_tools())
    def test_signature_validation_still_required(self):
        with self.assertRaises(P.AgentPlannerError):
            self.planner._parse_decision('{"decision_type":"tool","capability":"get_application_windows","arguments":{}}', self.tools.list_tools())
    def test_oversized_goal_stops_before_inference(self):
        self.state.goal.text = "x" * 1801
        with patch.object(self.planner, "_call_model") as model:
            with self.assertRaises(P.AgentPlannerError):
                self.planner.decide(self.state)
            model.assert_not_called()
        self.assertEqual(self.state.model_calls, 0)
    def test_registry_growth_is_not_silently_truncated(self):
        with self.assertRaises(P.AgentPlannerError):
            self.planner._build_prompt(self.state, self.tools.list_tools() * 300)
    def test_five_inspections_preserve_data_and_accounting(self):
        names = ["get_system_volume", "get_frontmost_application", "get_running_applications", "get_finder_windows", "get_frontmost_window"]
        data = [{"volume": 75, "muted": False}, {"name": "Safari"}, {"count": 180, "applications": [{"name": "App" + str(i), "path": "x" * 200} for i in range(180)]}, {"windows": [], "count": 0}, {"application": "Safari", "window": {"title": "Example"}}]
        tools = WindowTools(*(observation(data=x) for x in data))
        planner = P.AgentPlanner(tools)
        outputs = iter([json.dumps({"decision_type": "tool", "capability": name, "arguments": {}}) for name in names] + ['{"decision_type":"complete","message":"All five inspections returned results."}'])
        prompts = []
        def model(prompt):
            prompts.append(prompt)
            return next(outputs)
        state = M.AgentState(M.AgentGoal("Inspect volume, frontmost app, running apps, Finder windows and frontmost window."))
        with patch.object(planner, "_call_model", side_effect=model), contextlib.redirect_stdout(io.StringIO()):
            L.AgentLoop(planner, tools).run(state)
        self.assertEqual(state.status, M.AgentStatus.COMPLETED)
        self.assertEqual((state.tool_calls, state.model_calls, state.global_steps_used), (5, 6, 17))
        self.assertTrue(all(len(p) <= 18000 for p in prompts))
        self.assertEqual(len(state.steps[2].observation.data["applications"]), 180)
    def test_repair_consumes_model_and_global_budget(self):
        replies = [{"message": {"content": '{"decision_type":"correct"}'}}, {"message": {"content": '{"decision_type":"ask_user","message":"Which app?"}'}}]
        with patch.object(P.ollama, "chat", side_effect=replies) as chat, contextlib.redirect_stdout(io.StringIO()):
            L.AgentLoop(self.planner, self.tools).run(self.state)
        self.assertEqual(self.state.status, M.AgentStatus.WAITING_FOR_USER)
        self.assertEqual((self.state.model_calls, self.state.global_steps_used), (2, 3))
        self.assertEqual(chat.call_count, 2)
        self.assertTrue(all(isinstance(c.kwargs["format"], dict) for c in chat.call_args_list))


if __name__ == "__main__":
    unittest.main()