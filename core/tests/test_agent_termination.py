"""Deterministic Phase 6G checks: no Ollama calls or macOS actions.

Load the real models, progress policy, planner, recovery classifier and loop.
Replace Ollama transport and the tool adapter: no live model or Mac actions.
"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


def load_runtime():
    root = Path(__file__).resolve().parents[2]

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, root / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    class NoLiveDependency:
        def __init__(self, *args, **kwargs):
            raise AssertionError("This test must inject fake planner and tools")

    with patch.dict(sys.modules):
        for name in ("core", "core.agent", "core.actions", "core.capabilities"):
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        tools = types.ModuleType("core.agent.tool_adapter")
        tools.AgentToolAdapter = NoLiveDependency
        tools.AgentTool = object
        sys.modules[tools.__name__] = tools
        config = types.ModuleType("core.config")
        config.LLM_MODEL = "fake-test-model"
        config.OLLAMA_KEEP_ALIVE = "0"
        sys.modules[config.__name__] = config
        ollama = types.ModuleType("ollama")
        def no_live_call(*args, **kwargs):
            raise AssertionError("Live Ollama calls are forbidden in this suite")
        ollama.chat = no_live_call
        sys.modules["ollama"] = ollama
        models = load("core.agent.models", "core/agent/models.py")
        recovery = load("core.agent.recovery", "core/agent/recovery.py")
        cap_models = load("core.capabilities.models", "core/capabilities/models.py")
        registry_module = types.ModuleType("core.capabilities.registry")
        registry_module.registry = types.SimpleNamespace(get=lambda name: None)
        sys.modules[registry_module.__name__] = registry_module
        executor = load("core.capabilities.executor", "core/capabilities/executor.py")
        observer = load("core.agent.observer", "core/agent/observer.py")
        progress = load("core.agent.progress", "core/agent/progress.py")
        budgets = load("core.agent.budgets", "core/agent/budgets.py")
        load("core.actions.confirmation", "core/actions/confirmation.py")
        load("core.agent.confirmations", "core/agent/confirmations.py")
        planner = load("core.agent.planner", "core/agent/planner.py")
        loop = load("core.agent.loop", "core/agent/loop.py")
        persistence = load("core.agent.persistence", "core/agent/persistence.py")
        sessions = load("core.agent.sessions", "core/agent/sessions.py")
        loop.Recovery = recovery
        loop.CapModels = cap_models
        loop.Executor = executor
        loop.Observer = observer
        loop.Persistence = persistence
        loop.Sessions = sessions
    return models, loop, planner, progress, budgets


M, L, P, G, B = load_runtime()


class FakePlanner:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.calls = 0

    def decide(self, state):
        self.calls += 1
        if not self.decisions:
            raise AssertionError("Unexpected extra planning call")
        result = self.decisions.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeTools:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def execute(self, capability, arguments, *, confirmed=False):
        self.calls.append((capability, dict(arguments), confirmed))
        if not self.results:
            raise AssertionError("Unexpected extra tool call")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def decision(kind="tool", capability="get_system_volume", arguments=None):
    return M.AgentDecision(
        decision_type=M.AgentDecisionType(kind),
        capability=capability if kind == "tool" else None,
        arguments=arguments or {},
        message="Finished." if kind == "complete" else None,
    )


def observation(success=True, **kwargs):
    return M.AgentObservation(
        capability="get_system_volume", success=success,
        status=kwargs.pop("status", "success" if success else "failed"),
        **kwargs,
    )


class TerminationTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()

    def tearDown(self):
        self.redirect.__exit__(None, None, None)

    def state(self, **kwargs):
        return M.AgentState(goal=M.AgentGoal("Test goal"), **kwargs)

    def assert_stop(self, state, status, reason):
        self.assertEqual(state.status.value, status)
        self.assertEqual(state.termination_reason.value, reason)
        self.assertTrue(state.is_terminal)
        self.assertFalse(state.can_continue())
        self.assertIsNone(state.pending_confirmation)
        self.assertIsNone(state.pending_confirmation_step)
        json.dumps(state.termination_diagnostics())

    def paused(self, **kwargs):
        action = decision(arguments={"target": "example"})
        tools = FakeTools(observation(
            False, status="blocked", requires_confirmation=True,
            message="Allow this action?",
        ))
        planner = FakePlanner(action)
        loop = L.AgentLoop(planner=planner, tools=tools)
        state = loop.run(self.state(**kwargs))
        self.assertEqual(state.status, M.AgentStatus.WAITING_FOR_CONFIRMATION)
        return loop, state, planner, tools

    def test_complete_without_tools_preserves_existing_planner_contract(self):
        state = L.AgentLoop(FakePlanner(decision("complete")), FakeTools()).run(self.state())
        self.assert_stop(state, "completed", "goal_satisfied")
        self.assertIsNone(state.final_observation)

    def test_completed_preserves_snapshot_and_json_diagnostics(self):
        result = observation(data={"volume": 25, "path": Path("sample")})
        loop = L.AgentLoop(FakePlanner(decision(), decision("complete")), FakeTools(result))
        state = loop.run(self.state())
        self.assert_stop(state, "completed", "goal_satisfied")
        result.data["volume"] = 99
        self.assertEqual(state.final_observation.data["volume"], 25)
        self.assertEqual(state.termination_diagnostics()["final_observation"]["data"]["path"], "sample")
        record = json.loads(self.output.getvalue().split("[AGENT TERMINATION] ")[1])
        self.assertEqual(record["reason"], "goal_satisfied")

    def test_exhausted_initial_steps_become_terminal_without_planning(self):
        planner, tools = FakePlanner(), FakeTools()
        state = L.AgentLoop(planner, tools).run(self.state(max_steps=0))
        self.assert_stop(state, "limit_exceeded", "max_steps")
        self.assertEqual(planner.calls, 0)
        self.assertEqual(tools.calls, [])

    def test_existing_exhausted_history_is_not_reset(self):
        state = self.state(max_steps=1)
        state.steps.append(M.AgentStep(1, decision(), observation()))
        planner = FakePlanner(decision("complete"))
        state = L.AgentLoop(planner, FakeTools()).run(state)
        self.assert_stop(state, "completed", "goal_satisfied")
        self.assertEqual(planner.calls, 1)
        self.assertTrue(state.completion_check_used)

    def test_exhausted_initial_failures_become_terminal(self):
        planner = FakePlanner()
        state = L.AgentLoop(planner, FakeTools()).run(self.state(failure_count=3))
        self.assert_stop(state, "limit_exceeded", "max_failures")
        self.assertEqual(planner.calls, 0)

    def test_last_permitted_tool_is_preserved_when_limit_reached(self):
        tools = FakeTools(observation(data={"volume": 25}))
        planner = FakePlanner(decision(), decision())
        state = L.AgentLoop(planner, tools).run(self.state(max_steps=1))
        self.assert_stop(state, "limit_exceeded", "max_steps")
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(planner.calls, 2)
        self.assertTrue(state.final_observation.success)
        self.assertIn("not been undone", state.final_answer)

    def test_cumulative_failure_limit(self):
        tools = FakeTools(observation(False, retry_same_action=True), observation(False, retry_same_action=True))
        state = L.AgentLoop(FakePlanner(decision(), decision()), tools).run(self.state(max_failures=2))
        self.assert_stop(state, "limit_exceeded", "max_failures")
        self.assertEqual(state.failure_count, 2)

    def test_failed_tool_can_recover(self):
        tools = FakeTools(observation(False, retry_same_action=True), observation())
        state = L.AgentLoop(FakePlanner(decision(), decision(), decision("complete")), tools).run(self.state())
        self.assert_stop(state, "completed", "goal_satisfied")
        self.assertEqual(state.failure_count, 1)

    def test_planner_exception_preserves_previous_observation(self):
        state = L.AgentLoop(
            FakePlanner(decision(), RuntimeError("bad planner")),
            FakeTools(observation(data={"volume": 25})),
        ).run(self.state())
        self.assert_stop(state, "failed", "planner_error")
        self.assertEqual(state.final_observation.data, {"volume": 25})

    def test_invalid_decision_object(self):
        state = L.AgentLoop(FakePlanner(None), FakeTools()).run(self.state())
        self.assert_stop(state, "failed", "invalid_decision")

    def test_missing_capability_does_not_execute(self):
        tools = FakeTools()
        state = L.AgentLoop(FakePlanner(decision(capability=None)), tools).run(self.state())
        self.assert_stop(state, "failed", "invalid_decision")
        self.assertEqual(tools.calls, [])

    def test_unknown_decision_type_cannot_spin(self):
        invalid = M.AgentDecision(decision_type="not_a_type")
        state = L.AgentLoop(FakePlanner(invalid), FakeTools()).run(self.state())
        self.assert_stop(state, "failed", "invalid_decision")

    def test_unsupported_completion_after_failure(self):
        state = L.AgentLoop(
            FakePlanner(decision(), decision("complete")),
            FakeTools(observation(False)),
        ).run(self.state())
        self.assert_stop(state, "failed", "invalid_completion")

    def test_abort_without_available_action_is_blocked(self):
        state = L.AgentLoop(FakePlanner(decision("abort")), FakeTools()).run(self.state())
        self.assert_stop(state, "blocked", "no_safe_action")

    def test_abort_after_failure_is_failed(self):
        state = L.AgentLoop(FakePlanner(decision(), decision("abort")), FakeTools(observation(False))).run(self.state())
        self.assert_stop(state, "failed", "unrecoverable_failure")

    def test_abort_after_partial_success_is_blocked_with_progress_summary(self):
        state = L.AgentLoop(
            FakePlanner(decision(), decision("abort")),
            FakeTools(observation()),
        ).run(self.state())
        self.assert_stop(state, "blocked", "no_safe_action")
        self.assertIn("not been undone", state.final_answer)

    def test_permission_abort_is_blocked(self):
        state = L.AgentLoop(
            FakePlanner(decision(), decision("abort")),
            FakeTools(observation(False, recovery_type="permission_denied")),
        ).run(self.state())
        self.assert_stop(state, "blocked", "no_safe_action")

    def test_tool_exception_is_recorded(self):
        state = L.AgentLoop(FakePlanner(decision()), FakeTools(RuntimeError("adapter failed"))).run(self.state(max_failures=1))
        self.assert_stop(state, "limit_exceeded", "max_failures")
        self.assertIn("adapter failed", state.final_observation.error)

    def test_waiting_for_user_is_not_terminal(self):
        planner = FakePlanner(decision("ask_user"))
        loop = L.AgentLoop(planner, FakeTools())
        state = loop.run(self.state())
        self.assertEqual(state.status, M.AgentStatus.WAITING_FOR_USER)
        self.assertFalse(state.is_terminal)
        self.assertIsNone(state.termination_reason)
        loop.run(state)
        self.assertEqual(planner.calls, 1)

    def test_respond_also_waits(self):
        state = L.AgentLoop(FakePlanner(decision("respond")), FakeTools()).run(self.state())
        self.assertEqual(state.status, M.AgentStatus.WAITING_FOR_USER)
        self.assertFalse(state.is_terminal)

    def test_approval_resumes_exact_action_without_duplicate_step(self):
        loop, state, planner, tools = self.paused()
        run_id = state.run_id
        planner.decisions.append(decision("complete"))
        tools.results.append(observation())
        loop.confirm(state, True)
        self.assert_stop(state, "completed", "goal_satisfied")
        self.assertEqual(state.run_id, run_id)
        self.assertEqual(state.step_count, 1)
        self.assertEqual(tools.calls[0][:2], tools.calls[1][:2])
        self.assertTrue(tools.calls[1][2])
        with self.assertRaises(L.AgentLoopError):
            loop.confirm(state, True)
        self.assertEqual(len(tools.calls), 2)

    def test_rejection_is_cancelled_and_does_not_execute(self):
        loop, state, planner, tools = self.paused()
        loop.confirm(state, False)
        self.assert_stop(state, "cancelled", "confirmation_rejected")
        self.assertEqual(len(tools.calls), 1)

    def test_missing_pending_step_cannot_execute(self):
        loop, state, planner, tools = self.paused()
        state.pending_confirmation_step = 99
        loop.confirm(state, True)
        self.assert_stop(state, "failed", "invalid_confirmation")
        self.assertEqual(len(tools.calls), 1)

    def test_changed_pending_arguments_cannot_execute(self):
        loop, state, planner, tools = self.paused()
        state.pending_confirmation.arguments["target"] = "other"
        loop.confirm(state, True)
        self.assert_stop(state, "failed", "invalid_confirmation")
        self.assertEqual(len(tools.calls), 1)

    def test_failure_budget_checked_before_confirmed_execution(self):
        loop, state, planner, tools = self.paused()
        state.failure_count = state.max_failures
        loop.confirm(state, True)
        self.assert_stop(state, "limit_exceeded", "max_failures")
        self.assertEqual(len(tools.calls), 1)

    def test_confirmation_at_last_reserved_step_executes_once(self):
        loop, state, planner, tools = self.paused(max_steps=1)
        tools.results.append(observation())
        planner.decisions.append(decision("complete"))
        loop.confirm(state, True)
        self.assert_stop(state, "completed", "goal_satisfied")
        self.assertEqual(state.step_count, 1)
        self.assertEqual(len(tools.calls), 2)
        self.assertTrue(state.final_observation.success)

    def test_confirmed_failure_still_allows_recovery(self):
        loop, state, planner, tools = self.paused()
        tools.results.extend([observation(False, retry_same_action=True), observation()])
        planner.decisions.extend([decision(), decision("complete")])
        loop.confirm(state, True)
        self.assert_stop(state, "completed", "goal_satisfied")
        self.assertEqual(state.failure_count, 1)

    def test_repeated_confirmation_block_ends_run(self):
        loop, state, planner, tools = self.paused()
        tools.results.append(observation(False, status="blocked", requires_confirmation=True))
        loop.confirm(state, True)
        self.assert_stop(state, "blocked", "confirmation_still_blocked")

    def test_paused_cancellation(self):
        loop, state, planner, tools = self.paused()
        loop.cancel(state)
        self.assert_stop(state, "cancelled", "user_cancelled")
        self.assertEqual(len(tools.calls), 1)

    def test_terminal_rerun_is_a_no_op(self):
        planner, tools = FakePlanner(decision("complete")), FakeTools()
        loop = L.AgentLoop(planner, tools)
        state = loop.run(self.state())
        before = state.termination_diagnostics()
        loop.run(state)
        loop.cancel(state)
        self.assertEqual(before, state.termination_diagnostics())
        self.assertEqual(planner.calls, 1)

    def test_latest_observation_survives_incomplete_later_step(self):
        state = self.state()
        original = observation(data={"volume": 25})
        state.steps.extend([M.AgentStep(1, decision(), original), M.AgentStep(2, decision())])
        self.assertIs(state.latest_observation, original)
        state = L.AgentLoop(FakePlanner(RuntimeError("bad plan")), FakeTools()).run(state)
        self.assertEqual(state.final_observation.data, {"volume": 25})


class MetadataTools(FakeTools):
    def list_tools(self):
        return [types.SimpleNamespace(
            name=name, family="test", description="Test metadata",
            risk="low", requires_confirmation=False, parameters=[],
        ) for name in ("get_system_volume", "get_clipboard", "set_clipboard")]

    def validate_arguments(self, capability, arguments):
        return True, None


class ScriptedModelPlanner(P.AgentPlanner):
    def __init__(self, tools, *outputs):
        super().__init__(tool_adapter=tools)
        self.outputs = list(outputs)
        self.calls = 0

    def _call_model(self, prompt):
        self.calls += 1
        if not self.outputs:
            raise AssertionError("Unexpected extra model call")
        result = self.outputs.pop(0)
        if isinstance(result, str):
            return result
        return json.dumps({
            "decision_type": result.decision_type.value,
            "capability": result.capability,
            "arguments": result.arguments,
            "message": result.message,
            "reasoning_summary": result.reasoning_summary,
        })


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()

    def tearDown(self):
        self.redirect.__exit__(None, None, None)

    def state(self, **kwargs):
        kwargs.setdefault("max_failures", 10)
        return M.AgentState(goal=M.AgentGoal("Progress test"), **kwargs)

    def run_script(self, decisions, results, **kwargs):
        tools = FakeTools(*results)
        planner = FakePlanner(*decisions)
        state = L.AgentLoop(planner, tools).run(self.state(**kwargs))
        return state, planner, tools

    def test_repeat_successful_mutation_never_executes_twice(self):
        action = decision(capability="set_clipboard", arguments={"text": "hello"})
        state, planner, tools = self.run_script([action, action], [observation()])
        self.assertEqual(state.termination_reason.value, "repeated_decision")
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(state.step_count, 1)
        self.assertTrue(state.final_observation.success)

    def test_reordered_arguments_and_new_reasoning_do_not_disguise_repeat(self):
        first = decision(capability="move_window", arguments={"x": 1, "y": 2})
        again = decision(capability="move_window", arguments={"y": 2, "x": 1})
        again.reasoning_summary = "Different explanation"
        state, _, tools = self.run_script([first, again], [observation()])
        self.assertEqual(state.termination_reason.value, "repeated_decision")
        self.assertEqual(len(tools.calls), 1)

    def test_nonretryable_exact_failure_stops_before_dispatch(self):
        state, _, tools = self.run_script(
            [decision(), decision()], [observation(False, retry_same_action=False)],
        )
        self.assertEqual(state.termination_reason.value, "repeated_failed_strategy")
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(state.failure_count, 1)

    def test_unclassified_exact_failure_cannot_retry(self):
        state, _, tools = self.run_script([decision(), decision()], [observation(False)])
        self.assertEqual(state.termination_reason.value, "repeated_failed_strategy")
        self.assertEqual(len(tools.calls), 1)

    def test_read_only_retry_can_succeed_once(self):
        state, _, tools = self.run_script(
            [decision(), decision(), decision("complete")],
            [observation(False, retry_same_action=True), observation(data={"volume": 25})],
        )
        self.assertEqual(state.status.value, "completed")
        self.assertEqual(state.consecutive_failures, 0)
        self.assertEqual(state.failure_count, 1)
        self.assertEqual(len(tools.calls), 2)

    def test_third_identical_failed_read_is_blocked_even_with_other_limits_raised(self):
        state, _, tools = self.run_script(
            [decision()] * 3,
            [observation(False, retry_same_action=True)] * 2,
            max_consecutive_failures=10, max_no_progress_steps=10,
        )
        self.assertEqual(state.termination_reason.value, "repeated_failed_strategy")
        self.assertEqual(len(tools.calls), 2)

    def test_state_changing_timeout_does_not_retry(self):
        action = decision(capability="set_clipboard", arguments={"text": "hello"})
        state, _, tools = self.run_script(
            [action, action], [observation(False, retry_same_action=True, error="timeout")],
        )
        self.assertEqual(state.termination_reason.value, "repeated_failed_strategy")
        self.assertEqual(len(tools.calls), 1)

    def test_unknown_get_prefix_is_not_assumed_read_only(self):
        action = decision(capability="get_and_delete")
        state, _, tools = self.run_script([action, action], [observation(False, retry_same_action=True)])
        self.assertEqual(state.termination_reason.value, "repeated_failed_strategy")
        self.assertEqual(len(tools.calls), 1)

    def test_consecutive_limit_counts_distinct_failures(self):
        state, planner, tools = self.run_script(
            [decision(arguments={"n": 1}), decision(arguments={"n": 2})],
            [observation(False), observation(False)],
        )
        self.assertEqual(state.termination_reason.value, "max_consecutive_failures")
        self.assertEqual(state.consecutive_failures, 2)
        self.assertEqual(planner.calls, 2)

    def test_success_resets_consecutive_but_not_cumulative_failures(self):
        state, _, _ = self.run_script(
            [decision(arguments={"n": n}) for n in range(4)] + [decision("complete")],
            [observation(False), observation(data=1), observation(False), observation(data=2)],
        )
        self.assertEqual(state.status.value, "completed")
        self.assertEqual(state.failure_count, 2)
        self.assertEqual(state.consecutive_failures, 0)

    def test_unchanged_successful_inspection_is_no_progress(self):
        state, _, tools = self.run_script([decision()] * 4, [observation(data={"volume": 25})] * 4)
        self.assertEqual(state.termination_reason.value, "no_progress")
        self.assertEqual(state.no_progress_steps, 3)
        self.assertEqual(state.failure_count, 0)
        self.assertEqual(state.productive_steps, 1)
        self.assertEqual(len(tools.calls), 4)

    def test_alternating_read_cycle_is_detected(self):
        a, b = decision(), decision(capability="get_clipboard")
        state, _, tools = self.run_script(
            [a, b, a, b, a],
            [observation(data=25), observation(data="hello"), observation(data=25),
             observation(data="hello"), observation(data=25)],
        )
        self.assertEqual(state.termination_reason.value, "no_progress")
        self.assertEqual(len(tools.calls), 5)

    def test_changed_evidence_resets_progress_window(self):
        state, _, _ = self.run_script(
            [decision()] * 4 + [decision("complete")],
            [observation(data=1), observation(data=1), observation(data=2), observation(data=2)],
        )
        self.assertEqual(state.status.value, "completed")
        self.assertEqual(state.productive_steps, 2)
        self.assertEqual(state.no_progress_steps, 1)

    def test_changed_data_does_not_mean_goal_proof_but_allows_bounded_polling(self):
        state, _, tools = self.run_script([decision()] * 4, [observation(data=n) for n in range(3)], max_steps=3)
        self.assertEqual(state.termination_reason.value, "max_steps")
        self.assertEqual(state.productive_steps, 3)
        self.assertEqual(len(tools.calls), 3)

    def test_empty_data_counts_as_evidence(self):
        state, _, _ = self.run_script([decision(), decision("complete")], [observation(data=[])])
        self.assertEqual(state.productive_steps, 1)
        self.assertEqual(state.status.value, "completed")

    def test_changed_wrapper_message_does_not_disguise_same_data(self):
        state, _, _ = self.run_script(
            [decision()] * 4,
            [observation(data={"volume": 25}, message=str(n)) for n in range(4)],
        )
        self.assertEqual(state.termination_reason.value, "no_progress")

    def test_decision_ceiling_stops_before_additional_tool_call(self):
        state, _, tools = self.run_script(
            [decision()] * 4, [observation(data=25)] * 3,
            max_repeated_decisions=2, max_no_progress_steps=10,
        )
        self.assertEqual(state.termination_reason.value, "repeated_decision")
        self.assertEqual(len(tools.calls), 3)

    def test_pause_does_not_count_as_failure_or_clear_existing_streak(self):
        tools = FakeTools(observation(False), observation(False, status="blocked", requires_confirmation=True))
        planner = FakePlanner(decision(arguments={"n": 1}), decision(arguments={"n": 2}))
        loop = L.AgentLoop(planner, tools)
        state = loop.run(self.state())
        self.assertEqual(state.status.value, "waiting_for_confirmation")
        self.assertEqual(state.failure_count, 1)
        self.assertEqual(state.consecutive_failures, 1)
        self.assertEqual(state.no_progress_steps, 1)
        counts = dict(state.decisions_since_progress)
        loop.run(state)
        self.assertEqual(state.decisions_since_progress, counts)
        tools.results.append(observation(False))
        loop.confirm(state, True)
        self.assertEqual(state.termination_reason.value, "max_consecutive_failures")
        self.assertEqual(state.failure_count, 2)
        self.assertEqual(state.step_count, 2)
        self.assertEqual(sum(state.decisions_since_progress.values()), 2)

    def test_approved_success_resets_consecutive_failures(self):
        tools = FakeTools(observation(False), observation(False, status="blocked", requires_confirmation=True), observation(data=25))
        planner = FakePlanner(decision(arguments={"n": 1}), decision(arguments={"n": 2}), decision("complete"))
        loop = L.AgentLoop(planner, tools)
        state = loop.run(self.state())
        loop.confirm(state, True)
        self.assertEqual(state.status.value, "completed")
        self.assertEqual(state.failure_count, 1)
        self.assertEqual(state.consecutive_failures, 0)
        self.assertEqual(state.no_progress_steps, 0)

    def test_no_progress_limit_checked_before_approval_dispatch(self):
        tools = FakeTools(observation(False, status="blocked", requires_confirmation=True))
        loop = L.AgentLoop(FakePlanner(decision()), tools)
        state = loop.run(self.state())
        state.no_progress_steps = state.max_no_progress_steps
        loop.confirm(state, True)
        self.assertEqual(state.termination_reason.value, "no_progress")
        self.assertEqual(len(tools.calls), 1)

    def test_real_planner_repairs_repeated_mutation_to_completion(self):
        action = decision(capability="set_clipboard", arguments={"text": "hello"})
        tools = MetadataTools(observation())
        planner = ScriptedModelPlanner(tools, action, action, decision("complete"))
        state = L.AgentLoop(planner, tools).run(self.state())
        self.assertEqual(state.status.value, "completed")
        self.assertEqual(planner.calls, 3)
        self.assertEqual(len(tools.calls), 1)

    def test_real_planner_repeated_repair_has_specific_stop_reason(self):
        action = decision(capability="set_clipboard", arguments={"text": "hello"})
        tools = MetadataTools(observation())
        planner = ScriptedModelPlanner(tools, action, action, action)
        state = L.AgentLoop(planner, tools).run(self.state())
        self.assertEqual(state.termination_reason.value, "repeated_decision")
        self.assertEqual(len(tools.calls), 1)

    def test_real_planner_repairs_nonretryable_failure_with_different_strategy(self):
        tools = MetadataTools(observation(False, retry_same_action=False), observation(data="hello"))
        planner = ScriptedModelPlanner(tools, decision(), decision(), decision(capability="get_clipboard"), decision("complete"))
        state = L.AgentLoop(planner, tools).run(self.state())
        self.assertEqual(state.status.value, "completed")
        self.assertEqual(planner.calls, 4)
        self.assertEqual([call[0] for call in tools.calls], ["get_system_volume", "get_clipboard"])

    def test_real_planner_failed_strategy_reason_survives_repair(self):
        tools = MetadataTools(observation(False, retry_same_action=False))
        planner = ScriptedModelPlanner(tools, decision(), decision(), decision())
        state = L.AgentLoop(planner, tools).run(self.state())
        self.assertEqual(state.termination_reason.value, "repeated_failed_strategy")
        self.assertEqual(len(tools.calls), 1)

    def test_invalid_json_on_repair_is_reported_as_planner_error(self):
        action = decision(capability="set_clipboard", arguments={"text": "hello"})
        tools = MetadataTools(observation())
        planner = ScriptedModelPlanner(tools, action, action, "not json")
        state = L.AgentLoop(planner, tools).run(self.state())
        self.assertEqual(state.termination_reason.value, "planner_error")

    def test_shared_check_is_pure(self):
        state = self.state()
        before = state.termination_diagnostics()
        for _ in range(3):
            self.assertIsNone(G.check_action(state, decision()))
        self.assertEqual(before, state.termination_diagnostics())

    def test_diagnostics_include_counts_and_limits(self):
        state, _, _ = self.run_script([decision(), decision("complete")], [observation(data=25)])
        data = state.termination_diagnostics()
        self.assertEqual(data["productive_steps"], 1)
        self.assertEqual(data["distinct_evidence_count"], 1)
        self.assertEqual(data["limits"]["max_consecutive_failures"], 2)
        self.assertIn("max_no_progress_steps", data["limits"])
        json.dumps(data)


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.clock_patch = patch.object(B, "monotonic", self.clock)
        self.clock_patch.start()
        self.redirect = contextlib.redirect_stdout(io.StringIO())
        self.redirect.__enter__()

    def tearDown(self):
        self.redirect.__exit__(None, None, None)
        self.clock_patch.stop()

    def state(self, **kwargs):
        return M.AgentState(goal=M.AgentGoal("Budget test"), **kwargs)

    def assert_budget(self, state, reason):
        self.assertEqual(state.status.value, "limit_exceeded")
        self.assertEqual(state.termination_reason.value, reason)
        self.assertEqual(
            state.global_steps_used,
            state.decision_calls + state.model_calls + state.tool_calls,
        )
        self.assertLessEqual(state.global_steps_used, state.max_global_steps)
        self.assertLessEqual(state.model_calls, state.max_model_calls)

    def paused(self, **kwargs):
        tools = MetadataTools(observation(False, status="blocked", requires_confirmation=True))
        planner = ScriptedModelPlanner(tools, decision())
        loop = L.AgentLoop(planner, tools)
        state = loop.run(self.state(**kwargs))
        self.assertEqual(state.status.value, "waiting_for_confirmation")
        return state, loop, planner, tools

    def test_success_can_consume_exact_final_global_unit(self):
        tools = MetadataTools(observation(data=25))
        planner = ScriptedModelPlanner(tools, decision(), decision("complete"))
        state = L.AgentLoop(planner, tools).run(self.state(max_global_steps=5))
        self.assertEqual(state.status.value, "completed")
        self.assertEqual((state.decision_calls, state.model_calls, state.tool_calls), (2, 2, 1))
        self.assertEqual(state.global_steps_used, 5)

    def test_global_budget_checked_before_tool_dispatch(self):
        tools = MetadataTools()
        planner = ScriptedModelPlanner(tools, decision())
        state = L.AgentLoop(planner, tools).run(self.state(max_global_steps=2))
        self.assert_budget(state, "global_step_budget")
        self.assertEqual(tools.calls, [])
        self.assertEqual(state.step_count, 0)

    def test_zero_global_budget_does_no_work(self):
        planner, tools = FakePlanner(), FakeTools()
        state = L.AgentLoop(planner, tools).run(self.state(max_global_steps=0))
        self.assert_budget(state, "global_step_budget")
        self.assertEqual(planner.calls, 0)
        self.assertEqual(state.global_steps_used, 0)

    def test_global_budget_before_model_call(self):
        tools = MetadataTools()
        planner = ScriptedModelPlanner(tools)
        state = L.AgentLoop(planner, tools).run(self.state(max_global_steps=1))
        self.assert_budget(state, "global_step_budget")
        self.assertEqual(planner.calls, 0)
        self.assertEqual(state.model_calls, 0)

    def test_zero_model_budget_blocks_transport(self):
        tools = MetadataTools()
        planner = ScriptedModelPlanner(tools)
        state = L.AgentLoop(planner, tools).run(self.state(max_model_calls=0))
        self.assert_budget(state, "model_call_budget")
        self.assertEqual(planner.calls, 0)
        self.assertEqual(state.failure_count, 0)

    def test_repair_counts_as_another_model_attempt(self):
        tools = MetadataTools()
        planner = ScriptedModelPlanner(tools, "not json", decision("complete"))
        state = L.AgentLoop(planner, tools).run(self.state())
        self.assertEqual(state.status.value, "completed")
        self.assertEqual((state.decision_calls, state.model_calls, state.global_steps_used), (1, 2, 3))

    def test_model_budget_blocks_repair_before_transport(self):
        tools = MetadataTools()
        planner = ScriptedModelPlanner(tools, "not json", decision("complete"))
        state = L.AgentLoop(planner, tools).run(self.state(max_model_calls=1))
        self.assert_budget(state, "model_call_budget")
        self.assertEqual(planner.calls, 1)
        self.assertEqual(state.failure_count, 0)

    def test_global_budget_blocks_repair(self):
        tools = MetadataTools()
        planner = ScriptedModelPlanner(tools, "not json", decision("complete"))
        state = L.AgentLoop(planner, tools).run(self.state(max_global_steps=2))
        self.assert_budget(state, "global_step_budget")
        self.assertEqual(planner.calls, 1)

    def test_model_attempts_do_not_reset_between_planning_cycles(self):
        tools = MetadataTools(observation(data=25))
        planner = ScriptedModelPlanner(tools, "bad", decision(), "bad", decision("complete"))
        state = L.AgentLoop(planner, tools).run(self.state(max_model_calls=3))
        self.assert_budget(state, "model_call_budget")
        self.assertEqual(planner.calls, 3)
        self.assertEqual(state.final_observation.data, 25)

    def test_transport_failure_still_consumes_model_attempt(self):
        tools = MetadataTools()
        planner = P.AgentPlanner(tool_adapter=tools)
        with patch.object(planner, "_call_model", side_effect=RuntimeError("offline")):
            state = L.AgentLoop(planner, tools).run(self.state())
        self.assertEqual(state.termination_reason.value, "planner_error")
        self.assertEqual(state.model_calls, 1)
        self.assertEqual(state.global_steps_used, 2)

    def test_global_limit_also_bounds_non_model_planners(self):
        state = self.state(max_global_steps=3)
        tools = FakeTools(observation(data=25))
        planner = FakePlanner(decision(), decision("complete"))
        state = L.AgentLoop(planner, tools).run(state)
        self.assertEqual(state.status.value, "completed")
        self.assertEqual(state.global_steps_used, 3)
        self.assertEqual(state.model_calls, 0)

    def test_zero_time_budget_starts_no_planning(self):
        planner = FakePlanner()
        state = L.AgentLoop(planner, FakeTools()).run(self.state(max_runtime_seconds=0))
        self.assert_budget(state, "runtime_budget")
        self.assertEqual(planner.calls, 0)

    def test_slow_model_result_cannot_dispatch_tool(self):
        tools = MetadataTools()
        planner = ScriptedModelPlanner(tools, decision())
        original = planner._call_model
        def slow(prompt):
            self.clock.advance(6)
            return original(prompt)
        with patch.object(planner, "_call_model", side_effect=slow):
            state = L.AgentLoop(planner, tools).run(self.state(max_runtime_seconds=5))
        self.assert_budget(state, "runtime_budget")
        self.assertEqual(tools.calls, [])
        self.assertEqual(state.model_calls, 1)

    def test_deadline_boundary_is_inclusive(self):
        planner = FakePlanner(decision("complete"))
        original = planner.decide
        def slow(state):
            self.clock.advance(5)
            return original(state)
        with patch.object(planner, "decide", side_effect=slow):
            state = L.AgentLoop(planner, FakeTools()).run(self.state(max_runtime_seconds=5))
        self.assert_budget(state, "runtime_budget")

    def test_exception_after_deadline_reports_runtime_budget(self):
        planner = FakePlanner()
        def late_error(state):
            self.clock.advance(6)
            raise RuntimeError("planner returned late")
        with patch.object(planner, "decide", side_effect=late_error):
            state = L.AgentLoop(planner, FakeTools()).run(self.state(max_runtime_seconds=5))
        self.assert_budget(state, "runtime_budget")
        self.assertEqual(state.failure_count, 0)

    def test_slow_successful_tool_preserves_returned_observation(self):
        tools = FakeTools(observation(data={"volume": 25}))
        original = tools.execute
        def slow(*args, **kwargs):
            self.clock.advance(6)
            return original(*args, **kwargs)
        planner = FakePlanner(decision())
        with patch.object(tools, "execute", side_effect=slow):
            state = L.AgentLoop(planner, tools).run(self.state(max_runtime_seconds=5))
        self.assert_budget(state, "runtime_budget")
        self.assertEqual(state.final_observation.data, {"volume": 25})
        self.assertTrue(state.final_observation.success)
        self.assertEqual(state.productive_steps, 1)
        self.assertEqual(state.elapsed_seconds, 6)
        self.assertEqual(planner.calls, 1)

    def test_slow_failed_tool_preserves_error_before_budget_stop(self):
        tools = FakeTools()
        def slow_error(*args, **kwargs):
            self.clock.advance(6)
            raise RuntimeError("uncertain result")
        with patch.object(tools, "execute", side_effect=slow_error):
            state = L.AgentLoop(FakePlanner(decision()), tools).run(self.state(max_runtime_seconds=5))
        self.assert_budget(state, "runtime_budget")
        self.assertEqual(state.failure_count, 1)
        self.assertIn("uncertain result", state.final_observation.error)

    def test_late_confirmation_gate_terminates_instead_of_pausing(self):
        tools = FakeTools(observation(False, status="blocked", requires_confirmation=True))
        original = tools.execute
        def slow(*args, **kwargs):
            self.clock.advance(6)
            return original(*args, **kwargs)
        with patch.object(tools, "execute", side_effect=slow):
            state = L.AgentLoop(FakePlanner(decision()), tools).run(self.state(max_runtime_seconds=5))
        self.assert_budget(state, "runtime_budget")
        self.assertIsNone(state.pending_confirmation)
        self.assertEqual(state.failure_count, 0)
        self.assertEqual(state.final_observation.status, "blocked")

    def test_expired_approval_does_not_dispatch(self):
        state, loop, planner, tools = self.paused(max_runtime_seconds=5)
        self.clock.advance(6)
        loop.confirm(state, True)
        self.assert_budget(state, "runtime_budget")
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(state.tool_calls, 1)

    def test_rejection_after_deadline_remains_cancellation(self):
        state, loop, planner, tools = self.paused(max_runtime_seconds=5)
        self.clock.advance(6)
        loop.confirm(state, False)
        self.assertEqual(state.termination_reason.value, "confirmation_rejected")
        self.assertEqual(len(tools.calls), 1)

    def test_global_budget_applies_before_approved_dispatch(self):
        state, loop, planner, tools = self.paused(max_global_steps=3)
        loop.confirm(state, True)
        self.assert_budget(state, "global_step_budget")
        self.assertEqual(len(tools.calls), 1)

    def test_new_loop_object_cannot_reset_paused_run_budget(self):
        state, loop, planner, tools = self.paused()
        started = state.started_monotonic
        deadline = state.deadline_monotonic
        run_id = state.run_id
        self.clock.advance(4)
        tools.results.append(observation(data=25))
        next_planner = ScriptedModelPlanner(tools, decision("complete"))
        next_loop = L.AgentLoop(next_planner, tools)
        next_loop.confirm(state, True)
        self.assertEqual(state.status.value, "completed")
        self.assertEqual((state.run_id, state.started_monotonic, state.deadline_monotonic), (run_id, started, deadline))
        self.assertEqual((state.decision_calls, state.model_calls, state.tool_calls), (2, 2, 2))
        self.assertEqual(state.global_steps_used, 6)
        self.assertEqual(state.elapsed_seconds, 4)

    def test_model_budget_is_preserved_after_approval(self):
        state, loop, planner, tools = self.paused(max_model_calls=1)
        tools.results.append(observation(data=25))
        loop.confirm(state, True)
        self.assert_budget(state, "model_call_budget")
        self.assertEqual(state.model_calls, 1)
        self.assertEqual(state.tool_calls, 2)
        self.assertEqual(state.final_observation.data, 25)

    def test_confirmed_slow_tool_observation_is_preserved(self):
        state, loop, planner, tools = self.paused(max_runtime_seconds=5)
        tools.results.append(observation(data=25))
        original = tools.execute
        def slow(*args, **kwargs):
            self.clock.advance(6)
            return original(*args, **kwargs)
        with patch.object(tools, "execute", side_effect=slow):
            loop.confirm(state, True)
        self.assert_budget(state, "runtime_budget")
        self.assertEqual(state.final_observation.data, 25)
        self.assertEqual(state.step_count, 1)
        self.assertEqual(state.tool_calls, 2)

    def test_runtime_deadline_is_not_rebased_by_changing_config(self):
        state, loop, planner, tools = self.paused(max_runtime_seconds=5)
        state.max_runtime_seconds = 1000
        self.clock.advance(6)
        loop.confirm(state, True)
        self.assert_budget(state, "runtime_budget")
        self.assertEqual(state.runtime_budget_seconds, 5)

    def test_last_tool_slot_can_complete_with_real_planner(self):
        tools = MetadataTools(observation(data=25))
        planner = ScriptedModelPlanner(tools, decision(), decision("complete"))
        state = L.AgentLoop(planner, tools).run(self.state(max_steps=1))
        self.assertEqual(state.status.value, "completed")
        self.assertTrue(state.completion_check_used)
        self.assertEqual(state.step_count, 1)

    def test_final_completion_window_can_repair_without_executing_extra_tool(self):
        tools = MetadataTools(observation(data=25))
        planner = ScriptedModelPlanner(tools, decision(), decision(), decision("complete"))
        state = L.AgentLoop(planner, tools).run(self.state(max_steps=1))
        self.assertEqual(state.status.value, "completed")
        self.assertEqual(state.model_calls, 3)
        self.assertEqual(len(tools.calls), 1)

    def test_repaired_extra_tool_at_last_slot_stops_at_step_limit(self):
        tools = MetadataTools(observation(data=25))
        planner = ScriptedModelPlanner(tools, decision(), decision(), decision())
        state = L.AgentLoop(planner, tools).run(self.state(max_steps=1))
        self.assertEqual(state.termination_reason.value, "max_steps")
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(state.model_calls, 3)

    def test_final_completion_window_has_no_free_global_budget(self):
        tools = MetadataTools(observation(data=25))
        planner = ScriptedModelPlanner(tools, decision(), decision("complete"))
        state = L.AgentLoop(planner, tools).run(self.state(max_steps=1, max_global_steps=3))
        self.assert_budget(state, "global_step_budget")
        self.assertEqual(state.final_observation.data, 25)
        self.assertEqual(planner.calls, 1)

    def test_final_completion_window_has_no_free_model_attempt(self):
        tools = MetadataTools(observation(data=25))
        planner = ScriptedModelPlanner(tools, decision(), decision("complete"))
        state = L.AgentLoop(planner, tools).run(self.state(max_steps=1, max_model_calls=1))
        self.assert_budget(state, "model_call_budget")
        self.assertEqual(state.final_observation.data, 25)

    def test_final_completion_window_cannot_pause_and_reopen(self):
        tools = FakeTools(observation(data=25))
        planner = FakePlanner(decision(), decision("ask_user"))
        state = L.AgentLoop(planner, tools).run(self.state(max_steps=1))
        self.assertEqual(state.termination_reason.value, "max_steps")
        before = state.termination_diagnostics()
        L.AgentLoop(FakePlanner(), FakeTools()).run(state)
        self.assertEqual(before, state.termination_diagnostics())

    def test_terminal_time_and_counters_are_frozen(self):
        loop = L.AgentLoop(FakePlanner(decision("complete")), FakeTools())
        state = loop.run(self.state())
        before = state.termination_diagnostics()
        self.clock.advance(500)
        loop.run(state)
        loop.cancel(state)
        self.assertEqual(before, state.termination_diagnostics())

    def test_invalid_budget_configuration_is_rejected(self):
        for kwargs in (
            {"max_runtime_seconds": float("nan")},
            {"max_runtime_seconds": float("inf")},
            {"max_runtime_seconds": -1},
            {"max_runtime_seconds": True},
            {"max_global_steps": -1},
            {"max_model_calls": 1.5},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.state(**kwargs)



class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.clock_patch = patch.object(B, "monotonic", self.clock)
        self.clock_patch.start()
        self.redirect = contextlib.redirect_stdout(io.StringIO())
        self.redirect.__enter__()

    def tearDown(self):
        self.redirect.__exit__(None, None, None)
        self.clock_patch.stop()

    def paused(self, **kwargs):
        gate = observation(False, status="blocked", requires_confirmation=True)
        tools = FakeTools(gate, observation(data={"done": True}))
        planner = FakePlanner(decision(arguments={"target": "example"}), decision("complete"))
        loop = L.AgentLoop(planner, tools)
        state = loop.run(M.AgentState(goal=M.AgentGoal("Confirmation test"), **kwargs))
        self.assertEqual(state.status, M.AgentStatus.WAITING_FOR_CONFIRMATION)
        return loop, state, tools

    def events(self, state):
        return [event["event"] for event in state.confirmation_events]

    def test_confirmation_snapshot_and_explanation(self):
        loop, state, tools = self.paused()
        record = state.confirmation
        self.assertEqual(record.run_id, state.run_id)
        self.assertEqual(record.step_number, 1)
        self.assertEqual(json.loads(record.arguments_json), {"target": "example"})
        self.assertEqual(record.expires_monotonic, 160)
        self.assertIn("yes to approve, no to reject, or cancel", state.final_answer)
        self.assertIn("example", state.final_answer)
        self.assertEqual(self.events(state), ["requested"])
        json.dumps(state.termination_diagnostics())

    def test_confirmation_deadline_capped_by_runtime(self):
        loop, state, tools = self.paused(max_runtime_seconds=20)
        self.assertEqual(state.confirmation.expires_monotonic, 120)

    def test_approval_before_expiry_executes_saved_step_once(self):
        loop, state, tools = self.paused()
        self.clock.advance(59)
        loop.reply_to_confirmation(state, "  YES   please! ")
        self.assertEqual(state.status.value, "completed")
        self.assertEqual(len(state.steps), 1)
        self.assertEqual([c[2] for c in tools.calls], [False, True])
        self.assertEqual(self.events(state), ["requested", "approved", "execution_result"])
        self.assertIsNone(state.confirmation)

    def test_expiry_at_exact_deadline_blocks_execution(self):
        loop, state, tools = self.paused()
        self.clock.advance(60)
        loop.reply_to_confirmation(state, "yes")
        self.assertEqual(state.termination_reason.value, "confirmation_expired")
        self.assertEqual(state.status.value, "cancelled")
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(self.events(state), ["requested", "expired"])

    def test_runtime_exhaustion_takes_precedence(self):
        loop, state, tools = self.paused(max_runtime_seconds=20)
        self.clock.advance(20)
        loop.confirm(state, True)
        self.assertEqual(state.termination_reason.value, "runtime_budget")
        self.assertEqual(len(tools.calls), 1)

    def test_rejection_and_cancellation_have_distinct_reasons(self):
        for reply, reason, event in [("no", "confirmation_rejected", "rejected"),
                                     ("Cancel!", "user_cancelled", "cancelled"),
                                     ("never mind", "user_cancelled", "cancelled")]:
            with self.subTest(reply=reply):
                loop, state, tools = self.paused()
                loop.reply_to_confirmation(state, reply)
                self.assertEqual(state.termination_reason.value, reason)
                self.assertEqual(self.events(state), ["requested", event])
                self.assertEqual(len(tools.calls), 1)

    def test_ambiguous_reply_does_not_consume_confirmation(self):
        loop, state, tools = self.paused()
        record = state.confirmation
        with self.assertRaises(L.AgentLoopError):
            loop.reply_to_confirmation(state, "maybe")
        self.assertIs(state.confirmation, record)
        self.assertEqual(len(tools.calls), 1)

    def test_stale_tag_does_not_consume_current_request(self):
        loop, state, tools = self.paused()
        record = state.confirmation
        with self.assertRaises(L.AgentLoopError):
            loop.confirm(state, True, confirmation_id="old-request")
        self.assertIs(state.confirmation, record)
        self.assertEqual(self.events(state), ["requested", "stale_response"])
        self.assertEqual(len(tools.calls), 1)

    def test_matching_mutation_of_both_saved_decisions_is_detected(self):
        loop, state, tools = self.paused()
        state.pending_confirmation.arguments["target"] = "different"
        state.steps[0].decision.arguments["target"] = "different"
        loop.confirm(state, True)
        self.assertEqual(state.termination_reason.value, "invalid_confirmation")
        self.assertEqual(len(tools.calls), 1)

    def test_missing_snapshot_cannot_execute(self):
        loop, state, tools = self.paused()
        state.confirmation = None
        loop.confirm(state, True)
        self.assertEqual(state.termination_reason.value, "invalid_confirmation")
        self.assertEqual(len(tools.calls), 1)

    def test_changed_run_id_cannot_execute(self):
        loop, state, tools = self.paused()
        state.run_id = "another-run"
        loop.confirm(state, True)
        self.assertEqual(state.termination_reason.value, "invalid_confirmation")
        self.assertEqual(len(tools.calls), 1)

    def test_duplicate_sequential_approval_cannot_execute(self):
        loop, state, tools = self.paused()
        loop.confirm(state, True)
        with self.assertRaises(L.AgentLoopError):
            loop.confirm(state, True)
        self.assertEqual(len(tools.calls), 2)

    def test_superseded_request_cannot_execute_later(self):
        loop, state, tools = self.paused()
        loop.invalidate_confirmation(state)
        self.assertEqual(state.termination_reason.value, "confirmation_superseded")
        with self.assertRaises(L.AgentLoopError):
            loop.reply_to_confirmation(state, "yes")
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(self.events(state), ["requested", "invalidated"])

    def test_two_approvals_keep_run_budget_and_reject_old_id(self):
        gate = lambda: observation(False, status="blocked", requires_confirmation=True)
        tools = FakeTools(gate(), observation(data=1), gate(), observation(data=2))
        planner = FakePlanner(decision(arguments={"target": "one"}),
                              decision(arguments={"target": "two"}), decision("complete"))
        loop = L.AgentLoop(planner, tools)
        state = loop.run(M.AgentState(goal=M.AgentGoal("Two actions")))
        first = state.confirmation.confirmation_id
        run_id, deadline = state.run_id, state.deadline_monotonic
        used = state.global_steps_used
        loop.confirm(state, True, confirmation_id=first)
        second = state.confirmation.confirmation_id
        self.assertNotEqual(first, second)
        self.assertEqual(state.run_id, run_id)
        self.assertEqual(state.deadline_monotonic, deadline)
        self.assertGreater(state.global_steps_used, used)
        with self.assertRaises(L.AgentLoopError):
            loop.confirm(state, True, confirmation_id=first)
        self.assertEqual(len(tools.calls), 3)
        loop.confirm(state, True, confirmation_id=second)
        self.assertEqual(state.status.value, "completed")
        self.assertEqual(len(tools.calls), 4)
        self.assertEqual(len(state.steps), 2)

    def test_failure_after_approval_is_audited(self):
        loop, state, tools = self.paused()
        tools.results = [observation(False, message="Permission denied")]
        loop.planner.decisions = [decision("abort")]
        loop.confirm(state, True)
        self.assertEqual(self.events(state), ["requested", "approved", "execution_result"])
        self.assertFalse(state.confirmation_events[-1]["success"])
        self.assertFalse(state.final_observation.success)

    def test_invalid_confirmation_timeout_rejected(self):
        for value in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                M.AgentState(goal=M.AgentGoal("Test"), confirmation_timeout_seconds=value)



class EngineConfirmationTests(unittest.TestCase):
    """Exercise actual engine methods with fake dependencies; no live API."""
    def harness(self):
        import ast
        import time
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root / "core/engine.py").read_text())
        methods = [node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef)
                   and node.name in {"_chat", "_agent_response", "_agent_session_error"}]
        namespace = {"AgentState": M.AgentState, "AgentStatus": M.AgentStatus,
                     "ConversationMemory": object,
                     "AgentStorageError": L.Persistence.AgentStorageError,
                     "AgentLoopError": L.AgentLoopError,
                     "confirmation_intent": L.confirmation_intent, "time": time,
                     "route_request": lambda message, mode: types.SimpleNamespace(route="direct", message=message)}
        class FellThrough(Exception):
            pass
        def run_action(message):
            raise FellThrough(message)
        namespace["run_action"] = run_action
        exec(compile(ast.Module(body=methods, type_ignores=[]), "engine_methods", "exec"), namespace)
        class Harness:
            _chat = namespace["_chat"]
            _agent_session_error = namespace["_agent_session_error"]
            _agent_response = namespace["_agent_response"]
            def _get_memory(self, conversation_id): return []
            def _agent_tools_used(self, state): return []
            def _record_agent_turn(self, **kwargs): pass
        h = Harness()
        tools = FakeTools(observation(False, status="blocked", requires_confirmation=True),
                          observation(data=1))
        h._agent_loop = L.AgentLoop(FakePlanner(decision(), decision("complete")), tools)
        import tempfile
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        h._agent_sessions = L.Sessions.AgentSessions(directory.name, lambda: h._agent_loop)
        state = h._agent_sessions.start("chat", "Engine test")
        h._active_agent_states = {"chat": state}
        h._pending_actions = {}
        return h, state, tools, FellThrough

    def test_engine_busy_reply_does_not_fall_through_to_actions(self):
        with contextlib.redirect_stdout(io.StringIO()):
            h, state, tools, _ = self.harness()
            with h._agent_sessions.store.locked("chat"):
                result = h._chat("yes", "chat", run_id=state.run_id, confirmation_id=state.confirmation.confirmation_id)
        self.assertIn("in progress", result["answer"])
        self.assertEqual(len(tools.calls), 1)

    def test_engine_yes_resumes_and_clears_terminal_run(self):
        with contextlib.redirect_stdout(io.StringIO()):
            h, state, tools, _ = self.harness()
            h._active_agent_states.clear()  # Simulate lost in-memory cache.
            result = h._chat("yes", "chat", run_id=state.run_id, confirmation_id=state.confirmation.confirmation_id)
        state = h._agent_sessions.store.read("chat")[1]
        self.assertEqual(state.status.value, "completed")
        self.assertEqual(result["answer"], state.final_answer)
        self.assertNotIn("chat", h._active_agent_states)
        self.assertEqual(len(tools.calls), 2)

    def test_engine_no_and_cancel_use_distinct_lifecycle_reasons(self):
        for message, reason in [("no", "confirmation_rejected"), ("cancel", "user_cancelled")]:
            with self.subTest(message=message), contextlib.redirect_stdout(io.StringIO()):
                h, state, tools, _ = self.harness()
                h._chat(message, "chat", run_id=state.run_id, confirmation_id=state.confirmation.confirmation_id)
                state = h._agent_sessions.store.read("chat")[1]
                self.assertEqual(state.termination_reason.value, reason)
                self.assertEqual(len(tools.calls), 1)

    def test_engine_unrelated_message_invalidates_before_fallthrough(self):
        with contextlib.redirect_stdout(io.StringIO()):
            h, state, tools, FellThrough = self.harness()
            with self.assertRaises(FellThrough):
                h._chat("tell me a joke", "chat")
            state = h._agent_sessions.store.read("chat")[1]
            self.assertEqual(state.termination_reason.value, "confirmation_superseded")
            self.assertNotIn("chat", h._active_agent_states)
            with self.assertRaises(FellThrough):
                h._chat("yes", "chat")
            self.assertEqual(len(tools.calls), 1)



class PersistenceTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.clock = FakeClock()
        self.wall = FakeClock()
        self.wall.now = 10000
        self.clock_patch = patch.object(B, "monotonic", self.clock)
        self.wall_patch = patch.object(L.Persistence.time, "time", self.wall)
        self.clock_patch.start()
        self.wall_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.addCleanup(self.wall_patch.stop)
        self.redirect = contextlib.redirect_stdout(io.StringIO())
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def session(self, planner=None, tools=None):
        planner = planner or FakePlanner(decision(arguments={"target": "example"}), decision("complete"))
        tools = tools or FakeTools(observation(False, status="blocked", requires_confirmation=True), observation(data=1))
        return L.Sessions.AgentSessions(self.directory.name, lambda: L.AgentLoop(planner, tools)), tools

    def test_restart_preserves_entire_state_and_pending_action(self):
        session, tools = self.session()
        state = session.start("chat", "Do task")
        state.seen_evidence.add("evidence")
        state.decisions_since_progress["signature"] = 2
        state.steps[0].observation.data = {"path": Path("a"), "tuple": (1, 2), "set": {"x"}}
        session.store.save("chat", state, "idle")
        restored = session.store.read("chat")[1]
        self.assertEqual(L.Persistence.encode(state), L.Persistence.encode(restored))
        new, newtools = self.session(FakePlanner(decision("complete")), FakeTools(observation(data=2)))
        result = new.reply("chat", "yes", confirmation_id=state.confirmation.confirmation_id)
        self.assertEqual(result.status.value, "completed")
        self.assertEqual(newtools.calls, [("get_system_volume", {"target": "example"}, True)])
        self.assertEqual(result.run_id, state.run_id)
        self.assertEqual(result.tool_calls, state.tool_calls + 1)
        self.assertEqual(len(result.steps), 1)

    def test_restart_rebases_monotonic_without_resetting_deadlines(self):
        session, _ = self.session()
        original = session.start("chat", "Do task")
        self.wall.advance(10)
        self.clock.now = 3
        restored = session.store.read("chat")[1]
        self.assertEqual(restored.deadline_monotonic, 173)
        self.assertEqual(restored.confirmation.expires_monotonic, 53)
        self.assertEqual(restored.elapsed_seconds, 10)
        self.assertEqual(restored.global_steps_used, original.global_steps_used)

    def test_expired_approval_after_restart_never_dispatches(self):
        session, _ = self.session()
        session.start("chat", "Do task")
        self.wall.advance(61)
        new, tools = self.session()
        state = new.reply("chat", "yes")
        self.assertEqual(state.termination_reason.value, "confirmation_expired")
        self.assertEqual(tools.calls, [])

    def test_runtime_expired_during_shutdown(self):
        session, _ = self.session()
        session.start("chat", "Do task")
        self.wall.advance(181)
        new, tools = self.session()
        state = new.reply("chat", "yes")
        self.assertEqual(state.termination_reason.value, "runtime_budget")
        self.assertEqual(tools.calls, [])

    def test_wall_clock_rollback_cannot_extend_approval(self):
        session, _ = self.session()
        session.start("chat", "Do task")
        self.wall.advance(-10)
        new, tools = self.session()
        state = new.reply("chat", "yes")
        self.assertEqual(state.status.value, "limit_exceeded")
        self.assertEqual(tools.calls, [])

    def test_repeated_load_save_does_not_grant_extra_time(self):
        session, _ = self.session()
        session.start("chat", "Do task")
        for _ in range(3):
            self.wall.advance(15)
            self.clock.advance(15)
            state = session.store.read("chat")[1]
            session.store.save("chat", state, "idle")
        self.assertEqual(state.confirmation.expires_monotonic - self.clock(), 15)
        self.assertEqual(state.deadline_monotonic - self.clock(), 135)

    def test_snapshot_unknown_version_and_corruption_fail_closed(self):
        session, _ = self.session()
        state = session.start("chat", "Do task")
        payload = json.loads(L.Persistence.dumps_state(state))
        payload["version"] = 999
        for text in ("not json", json.dumps(payload)):
            with self.subTest(text=text), self.assertRaises(L.Persistence.AgentStorageError):
                L.Persistence.loads_state(text)

    def test_unsupported_values_fail_without_string_coercion(self):
        state = M.AgentState(goal=M.AgentGoal("Test"))
        state.final_observation = observation(data=object())
        with self.assertRaises(L.Persistence.AgentStorageError):
            L.Persistence.dumps_state(state)

    def test_storage_failure_before_approval_dispatch_does_not_execute(self):
        session, tools = self.session()
        session.start("chat", "Do task")
        with patch.object(session.store, "save", side_effect=L.Persistence.AgentStorageError("disk full")):
            with self.assertRaises(L.Persistence.AgentStorageError):
                session.reply("chat", "yes")
        self.assertEqual(len(tools.calls), 1)

    def test_checkpoint_failure_before_tool_dispatch_prevents_call(self):
        session, tools = self.session()
        session.start("chat", "Do task")
        save = session.store.save
        calls = []
        def failing_save(conversation, state, phase):
            calls.append(phase)
            if len(calls) == 2:
                raise L.Persistence.AgentStorageError("disk full")
            return save(conversation, state, phase)
        with patch.object(session.store, "save", side_effect=failing_save):
            with self.assertRaises(L.Persistence.AgentStorageError):
                session.reply("chat", "yes")
        self.assertEqual(len(tools.calls), 1)
        recovered = session.reply("chat", "yes")
        self.assertEqual(recovered.termination_reason.value, "interrupted_run")
        self.assertEqual(len(tools.calls), 1)

    def test_interrupted_busy_run_preserves_progress_without_replay(self):
        session, tools = self.session()
        state = session.start("chat", "Do task")
        state.steps.insert(0, M.AgentStep(0, decision(), observation(data="saved progress")))
        session.store.save("chat", state, "busy")
        new, newtools = self.session()
        recovered = new.reply("chat", "yes")
        self.assertEqual(recovered.status.value, "blocked")
        self.assertEqual(recovered.termination_reason.value, "interrupted_run")
        self.assertEqual(recovered.steps[0].observation.data, "saved progress")
        self.assertEqual(newtools.calls, [])
        self.assertIsNone(new.reply("chat", "yes"))

    def test_concurrent_sessions_cannot_approve_together(self):
        import threading
        session, _ = self.session()
        session.start("chat", "Do task")
        entered, release = threading.Event(), threading.Event()
        class BlockingTools(FakeTools):
            def execute(self, *args, **kwargs):
                entered.set()
                if not release.wait(5): raise AssertionError("Test timed out")
                return super().execute(*args, **kwargs)
        tools = BlockingTools(observation(data=1))
        first, _ = self.session(FakePlanner(decision("complete")), tools)
        second, other_tools = self.session()
        results = []
        def approve():
            try: results.append(first.reply("chat", "yes"))
            except Exception as exc: results.append(exc)
        thread = threading.Thread(target=approve)
        thread.start()
        try:
            self.assertTrue(entered.wait(3))
            with self.assertRaises(L.Persistence.AgentBusyError):
                second.reply("chat", "yes")
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0].status.value, "completed")
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(other_tools.calls, [])

    def test_process_crash_during_action_releases_lock_without_replay(self):
        import multiprocessing
        import os
        session, _ = self.session()
        state = session.start("chat", "Do task")
        marker = Path(self.directory.name) / "executed"
        def crash():
            class CrashTools:
                def execute(self, *args, **kwargs):
                    marker.write_text("action began")
                    os._exit(23)
            child = L.Sessions.AgentSessions(self.directory.name,
                lambda: L.AgentLoop(FakePlanner(), CrashTools()))
            child.reply("chat", "yes")
        process = multiprocessing.get_context("fork").Process(target=crash)
        process.start()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join()
            self.fail("Crash test child timed out")
        self.assertEqual(process.exitcode, 23)
        self.assertTrue(marker.exists())
        new, tools = self.session()
        restored = new.reply("chat", "yes")
        self.assertEqual(restored.termination_reason.value, "interrupted_run")
        self.assertEqual(restored.tool_calls, state.tool_calls + 1)
        self.assertEqual(tools.calls, [])
        self.assertEqual([e["event"] for e in restored.confirmation_events][:2], ["requested", "approved"])

    def test_terminal_snapshot_is_frozen_across_restart(self):
        session, _ = self.session()
        session.start("chat", "Do task")
        state = session.reply("chat", "no")
        before = state.termination_diagnostics()
        self.wall.advance(999)
        self.clock.now = 1
        self.assertEqual(before, session.store.read("chat")[1].termination_diagnostics())

    def test_new_goal_cannot_overwrite_pending_run_after_restart(self):
        session, _ = self.session()
        state = session.start("chat", "Original")
        new, tools = self.session()
        result = new.start("chat", "Replacement")
        self.assertEqual(result.run_id, state.run_id)
        self.assertEqual(result.goal.text, "Original")
        self.assertEqual(tools.calls, [])

    def test_new_goal_after_terminal_gets_new_budget(self):
        session, _ = self.session()
        original = session.start("chat", "Original")
        session.reply("chat", "no")
        new, _ = self.session()
        result = new.start("chat", "Next")
        self.assertNotEqual(result.run_id, original.run_id)
        self.assertEqual(result.global_steps_used, original.global_steps_used)

    def test_stale_tag_does_not_mark_current_request_busy(self):
        session, tools = self.session()
        state = session.start("chat", "Original")
        with self.assertRaises(L.AgentLoopError):
            session.reply("chat", "yes", confirmation_id="wrong")
        phase, restored = session.store.read("chat")
        self.assertEqual(phase, "idle")
        self.assertEqual(restored.confirmation.confirmation_id, state.confirmation.confirmation_id)
        self.assertEqual(len(tools.calls), 1)



class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.redirect = contextlib.redirect_stdout(io.StringIO())
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def run_script(self, actions, results, **limits):
        tools = FakeTools(*results)
        loop = L.AgentLoop(FakePlanner(*actions), tools)
        state = loop.run(M.AgentState(goal=M.AgentGoal("Recovery test"), **limits))
        return state, tools

    def test_failure_category_matrix(self):
        cases = [
            ("PermissionError: operation not permitted", "permission_denied", False),
            ("FileNotFoundError: no such file", "not_found", False),
            ("FileExistsError: already exists", "already_exists", False),
            ("Invalid capability arguments.", "invalid_argument", False),
            ("No active window", "invalid_state", False),
            ("Capability 'x' is not currently available.", "tool_unavailable", False),
            ("application is not running", "application_unavailable", False),
            ("TimeoutError:", "timeout", True),
            ("ConnectionError:", "transient", True),
            ("Unsupported operation", "non_retryable", False),
            ("unexplained failure", "unknown", False),
        ]
        for error, category, retry in cases:
            with self.subTest(error=error):
                obs = L.Recovery.annotate_failure(observation(False, error=error))
                self.assertEqual(obs.recovery_type, category)
                self.assertEqual(obs.retry_same_action, retry)
                self.assertTrue(obs.recovery_guidance)

    def test_permission_overrides_timeout_and_retry_hint(self):
        obs = observation(False, error="Permission denied after timeout", retry_same_action=True)
        L.Recovery.annotate_failure(obs)
        self.assertEqual(obs.recovery_type, "permission_denied")
        self.assertFalse(obs.retry_same_action)

    def test_explicit_no_retry_remains_no_retry_for_timeout(self):
        obs = observation(False, error="timeout", retry_same_action=False)
        self.assertFalse(L.Recovery.annotate_failure(obs).retry_same_action)

    def test_unknown_get_prefix_not_retryable(self):
        obs = M.AgentObservation("get_and_delete", False, "failed", error="timeout")
        self.assertFalse(L.Recovery.annotate_failure(obs).retry_same_action)

    def test_exception_type_retained_even_with_empty_message(self):
        for exc, category in [(PermissionError(), "permission_denied"),
                              (FileNotFoundError(), "not_found"),
                              (TimeoutError(), "timeout"),
                              (ConnectionError(), "transient")]:
            with self.subTest(category=category):
                state, tools = self.run_script([decision(), decision("abort")], [exc])
                self.assertEqual(state.final_observation.recovery_type, category)
                self.assertIn(type(exc).__name__, state.final_observation.error)

    def test_returned_failure_is_normalized_before_replanning(self):
        state, tools = self.run_script([decision(), decision(), decision("complete")],
                                      [observation(False, error="timed out"), observation(data=5)])
        self.assertEqual(state.status.value, "completed")
        self.assertEqual(state.steps[0].observation.recovery_type, "timeout")
        self.assertTrue(state.steps[0].observation.retry_same_action)
        self.assertEqual(state.failure_count, 1)
        self.assertEqual(state.consecutive_failures, 0)

    def test_normalization_uses_dispatched_capability(self):
        action = decision(capability="set_clipboard", arguments={"text": "x"})
        state, tools = self.run_script([action, decision("abort")],
                                      [observation(False, error="timeout", retry_same_action=True)])
        self.assertEqual(state.final_observation.capability, "set_clipboard")
        self.assertFalse(state.final_observation.retry_same_action)

    def test_one_read_retry_then_stop_even_with_large_budgets(self):
        state, tools = self.run_script([decision()] * 3,
                                      [observation(False, error="timeout")] * 2,
                                      max_failures=10, max_consecutive_failures=10,
                                      max_no_progress_steps=10)
        self.assertEqual(state.termination_reason.value, "repeated_failed_strategy")
        self.assertEqual(len(tools.calls), 2)

    def test_global_budget_prevents_recovery_dispatch(self):
        state, tools = self.run_script([decision(), decision()],
                                      [observation(False, error="temporarily unavailable")],
                                      max_global_steps=2)
        self.assertEqual(state.termination_reason.value, "global_step_budget")
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(state.failure_count, 1)

    def test_missing_resource_can_recover_with_corrected_target(self):
        first = decision(arguments={"path": "missing"})
        corrected = decision(arguments={"path": "existing"})
        state, tools = self.run_script([first, corrected, decision("complete")],
                                      [observation(False, error="no such file"), observation(data="found")])
        self.assertEqual(state.status.value, "completed")
        self.assertEqual(len(tools.calls), 2)
        self.assertEqual(state.steps[0].observation.recovery_type, "not_found")

    def test_invalid_arguments_can_recover_with_changed_arguments(self):
        state, tools = self.run_script(
            [decision(arguments={"x": 1}), decision(arguments={"value": 1}), decision("complete")],
            [observation(False, error="unexpected keyword argument x"), observation(data=1)])
        self.assertEqual(state.status.value, "completed")
        self.assertEqual(state.steps[0].observation.recovery_type, "invalid_argument")

    def test_timeout_cannot_be_bypassed_by_different_mutation(self):
        actions = [decision(capability="set_clipboard", arguments={"text": "a"}),
                   decision(capability="create_folder", arguments={"name": "b"})]
        state, tools = self.run_script(actions, [observation(False, error="timeout")])
        self.assertEqual(state.termination_reason.value, "no_safe_action")
        self.assertEqual(len(tools.calls), 1)

    def test_permission_cannot_be_bypassed_by_another_mutation(self):
        actions = [decision(), decision(capability="set_clipboard", arguments={"text": "x"})]
        state, tools = self.run_script(actions, [observation(False, error="permission denied")])
        self.assertEqual(state.termination_reason.value, "no_safe_action")
        self.assertEqual(len(tools.calls), 1)

    def test_uncertain_mutation_allows_inspection_but_not_later_mutation(self):
        actions = [decision(capability="set_clipboard", arguments={"text": "a"}),
                   decision(capability="get_clipboard"),
                   decision(capability="set_clipboard", arguments={"text": "b"})]
        state, tools = self.run_script(actions, [observation(False, error="unknown issue"), observation(data="a")])
        self.assertEqual(state.termination_reason.value, "no_safe_action")
        self.assertEqual(len(tools.calls), 2)
        self.assertEqual(state.final_observation.data, "a")

    def test_real_planner_repairs_uncertain_mutation_to_inspection(self):
        action = decision(capability="set_clipboard", arguments={"text": "a"})
        alternative = decision(capability="set_clipboard", arguments={"text": "b"})
        tools = MetadataTools(observation(False, error="timeout"), observation(data="a"))
        planner = ScriptedModelPlanner(tools, action, alternative,
                                      decision(capability="get_clipboard"), decision("complete"))
        state = L.AgentLoop(planner, tools).run(M.AgentState(goal=M.AgentGoal("Set clipboard to a")))
        self.assertEqual(state.status.value, "completed")
        self.assertEqual(planner.calls, 4)
        self.assertEqual([x[0] for x in tools.calls], ["set_clipboard", "get_clipboard"])
        self.assertEqual(state.model_calls, 4)

    def test_real_planner_repeated_unsafe_repair_stops(self):
        action = decision(capability="set_clipboard", arguments={"text": "a"})
        alternative = decision(capability="set_clipboard", arguments={"text": "b"})
        tools = MetadataTools(observation(False, error="timeout"))
        planner = ScriptedModelPlanner(tools, action, alternative, alternative)
        state = L.AgentLoop(planner, tools).run(M.AgentState(goal=M.AgentGoal("Test")))
        self.assertEqual(state.termination_reason.value, "no_safe_action")
        self.assertEqual(len(tools.calls), 1)

    def test_failure_guidance_is_in_real_planner_history(self):
        state, _ = self.run_script([decision(), decision("abort")],
                                   [observation(False, error="application is not running")])
        planner = ScriptedModelPlanner(MetadataTools())
        history = planner._format_history(state)
        self.assertIn("application_unavailable", history)
        self.assertIn("make it available", history)

    def test_abort_explains_cause_and_preserves_partial_progress(self):
        state, tools = self.run_script([decision(), decision(arguments={"x": 2}), decision("abort")],
                                      [observation(data=1), observation(False, error="no such file: report.txt")])
        self.assertIn("report.txt", state.final_answer)
        self.assertIn("not been undone", state.final_answer)
        self.assertEqual(state.steps[0].observation.data, 1)
        self.assertEqual(state.termination_diagnostics()["recovery"]["failed_steps"][0]["category"], "not_found")

    def test_planner_failure_preserves_tool_observation(self):
        state, tools = self.run_script([decision(), RuntimeError("planner offline")], [observation(data=42)])
        self.assertEqual(state.termination_reason.value, "planner_error")
        self.assertEqual(state.final_observation.data, 42)
        self.assertTrue(state.termination_diagnostics()["recovery"]["planner_failure"])

    def test_confirmation_gate_is_not_recovery_failure(self):
        state, _ = self.run_script([decision()], [observation(False, status="blocked", requires_confirmation=True)])
        self.assertEqual(state.failure_count, 0)
        self.assertEqual(state.termination_diagnostics()["recovery"]["failed_steps"], [])
        self.assertEqual(state.latest_observation.recovery_type, "confirmation_required")

    def test_saved_approval_rechecks_recovery_policy_before_dispatch(self):
        action = decision(capability="set_clipboard", arguments={"text": "x"})
        tools = FakeTools(observation(False, status="blocked", requires_confirmation=True))
        loop = L.AgentLoop(FakePlanner(action), tools)
        state = loop.run(M.AgentState(goal=M.AgentGoal("Test")))
        # A legacy snapshot may have reached approval before this policy existed.
        failure = L.Recovery.annotate_failure(observation(False, error="permission denied"))
        state.steps.insert(0, M.AgentStep(0, decision(), failure))
        loop.confirm(state, True)
        self.assertEqual(state.termination_reason.value, "no_safe_action")
        self.assertEqual(len(tools.calls), 1)

    def test_recovery_fields_survive_version_one_persistence(self):
        state, _ = self.run_script([decision(), decision("abort")], [observation(False, error="no active window")])
        restored = L.Persistence.loads_state(L.Persistence.dumps_state(state))
        self.assertEqual(restored.termination_diagnostics(), state.termination_diagnostics())
        self.assertEqual(restored.final_observation.recovery_type, "invalid_state")

    def test_executor_observer_preserve_typed_failure_end_to_end(self):
        for exc, expected in [(TimeoutError(), "timeout"), (PermissionError(), "permission_denied"),
                              (FileNotFoundError(), "not_found"), (NotImplementedError(), "non_retryable")]:
            with self.subTest(expected=expected):
                def handler(): raise exc
                cap = L.CapModels.Capability("get_system_volume", "test", "test", handler)
                with patch.object(L.Executor.registry, "get", return_value=cap):
                    result = L.Executor.execute_capability("get_system_volume")
                obs = L.Observer.observe_execution(result)
                self.assertEqual(obs.recovery_type, expected)
                self.assertIn(type(exc).__name__, obs.error)

    def test_executor_missing_capability_is_tool_unavailable(self):
        with patch.object(L.Executor.registry, "get", return_value=None):
            result = L.Executor.execute_capability("missing_tool")
        obs = L.Observer.observe_execution(result)
        self.assertEqual(obs.recovery_type, "tool_unavailable")
        self.assertFalse(obs.retry_same_action)

    def test_executor_invalid_arguments_do_not_call_handler(self):
        calls = []
        def handler(required): calls.append(required)
        cap = L.CapModels.Capability("get_system_volume", "test", "test", handler)
        with patch.object(L.Executor.registry, "get", return_value=cap):
            result = L.Executor.execute_capability("get_system_volume", {"wrong": 1})
        obs = L.Observer.observe_execution(result)
        self.assertEqual(obs.recovery_type, "invalid_argument")
        self.assertEqual(calls, [])


class LiveReadbackRegressionTests(unittest.TestCase):
    def run_loop(self, decisions, results, **limits):
        state = M.AgentState(M.AgentGoal("Read volume and frontmost application"), **limits)
        with contextlib.redirect_stdout(io.StringIO()):
            L.AgentLoop(FakePlanner(*decisions), FakeTools(*results)).run(state)
        return state

    def test_generic_completion_still_reports_both_observations(self):
        end = decision("complete")
        end.message = "The requested task is complete."
        state = self.run_loop(
            [decision(), decision(capability="get_frontmost_application"), end],
            [observation(data={"volume": 87, "muted": False}), observation(data={"name": "Code"})],
        )
        self.assertEqual(state.status, M.AgentStatus.COMPLETED)
        self.assertIn("87%", state.final_answer)
        self.assertIn("not muted", state.final_answer)
        self.assertIn("Code", state.final_answer)
        diagnostics = state.termination_diagnostics()
        self.assertEqual(len(diagnostics["inspection_results"]), 2)
        self.assertEqual([x["capability"] for x in diagnostics["step_trace"]],
                         ["get_system_volume", "get_frontmost_application"])
        self.assertEqual(state.global_steps_used, 5)  # fake planner: 3 decisions + 2 tools

    def test_no_progress_keeps_observed_answer_without_claiming_completed(self):
        state = self.run_loop([decision(), decision()],
            [observation(data={"volume": 87, "muted": False}) for _ in range(2)],
            max_no_progress_steps=1)
        self.assertEqual(state.termination_reason, M.AgentTerminationReason.NO_PROGRESS)
        self.assertEqual(state.status, M.AgentStatus.BLOCKED)
        self.assertIn("87%", state.final_answer)
        self.assertEqual(len(state.inspection_results()), 1)

    def test_invalid_or_missing_snapshot_data_is_not_invented(self):
        for data in ({"volume": True, "muted": False}, {"volume": 101, "muted": False},
                     {"volume": float("nan"), "muted": False}, {"volume": 50},
                     {"volume": 50, "muted": "false"}, None):
            with self.subTest(data=data):
                state = self.run_loop([decision(), decision("complete")], [observation(data=data)])
                self.assertEqual(state.inspection_results(), [])

    def test_zero_volume_and_muted_are_preserved(self):
        state = self.run_loop([decision(), decision("complete")],
                             [observation(data={"volume": 0, "muted": True})])
        self.assertIn("0%; muted.", state.final_answer)

    def test_latest_failed_inspection_does_not_publish_an_old_sample(self):
        state = self.run_loop([decision(), decision(), decision("abort")],
                             [observation(data={"volume": 20, "muted": False}), observation(False)])
        self.assertEqual(state.inspection_results(), [])

    def test_readback_does_not_expose_arbitrary_tool_data(self):
        state = self.run_loop([decision(capability="get_clipboard"), decision("complete")],
                             [observation(data={"text": "private clipboard content"})])
        self.assertEqual(state.inspection_results(), [])
        self.assertNotIn("private clipboard content", state.final_answer)
        self.assertNotIn("data", state.termination_diagnostics()["step_trace"][0])

    def test_real_planner_receives_observed_value_after_capability_list(self):
        tools = MetadataTools(observation(data={"volume": 63, "muted": False}))
        planner = P.AgentPlanner(tools)
        prompts = []
        def model(prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                return '{"decision_type":"tool","capability":"get_system_volume","arguments":{}}'
            self.assertIn('"volume": 63', prompt.split("FINAL CHECK", 1)[1])
            return '{"decision_type":"complete","message":"Volume is 63%, not muted."}'
        state = M.AgentState(M.AgentGoal("How loud is my Mac?"))
        with patch.object(planner, "_call_model", side_effect=model), contextlib.redirect_stdout(io.StringIO()):
            L.AgentLoop(planner, tools).run(state)
        self.assertEqual(state.status, M.AgentStatus.COMPLETED)
        self.assertEqual(state.model_calls, 2)
        self.assertEqual(len(tools.calls), 1)


if __name__ == "__main__":
    unittest.main()
