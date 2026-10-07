"""6H integration checks: real engine/session/router, fake external systems."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

# Works with discovery by path or package.
try:
    from .test_agent_termination import M, L, P, B, FakePlanner, FakeTools, decision, observation
except ImportError:
    from test_agent_termination import M, L, P, B, FakePlanner, FakeTools, decision, observation

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Memory:
    def __init__(self, conversation):
        self.conversation = conversation
        self.messages = []
    def get_messages(self, limit=20, include_metadata=False):
        return self.messages[-limit:]
    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})
    def add_assistant_message(self, message, **metadata):
        self.messages.append({"role": "assistant", "content": message, **metadata})
    def get_full_history(self): return self.messages
    def clear(self): self.messages.clear()


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        def module(name, **attrs):
            obj = types.ModuleType(name)
            obj.__path__ = []
            obj.__dict__.update(attrs)
            sys.modules[name] = obj
            return obj
        module("core")
        module("core.actions")
        module("core.agent", AgentGoal=M.AgentGoal, AgentState=M.AgentState,
               AgentStatus=M.AgentStatus, AgentLoop=L.AgentLoop)
        module("core.agent.sessions", AgentSessions=L.Sessions.AgentSessions)
        module("core.agent.persistence", AgentStorageError=L.Persistence.AgentStorageError)
        module("core.agent.loop", AgentLoopError=L.AgentLoopError)
        module("core.actions.confirmation", confirmation_intent=L.confirmation_intent)
        self.direct_calls, self.chat_calls = [], []
        self.direct_result = types.SimpleNamespace(status="SUCCESS", capability="open_application", message="Opened", requires_confirmation=False)
        def direct(message):
            self.direct_calls.append(message)
            return self.direct_result
        def ask(message, memory, action_context=None):
            self.chat_calls.append((message, memory, action_context))
            memory.add_user_message(message)
            memory.add_assistant_message("RAG answer")
            return "RAG answer", [{"source": "docs", "text": "evidence"}], "rewritten", ["search_docs"], {}
        module("core.actions.runner", run_action=direct, confirm_action=lambda pending: pending)
        module("core.actions.responses", **{name: lambda result: result.message for name in (
            "format_action_success", "format_action_failure", "format_action_confirmation", "format_action_cancelled")})
        module("core.memory", ConversationMemory=Memory)
        module("core.rag", ask=ask)
        def plan(text):
            understood = text.lower() in {"open safari", "set volume to 30", 'create folder "rock and roll"'}
            return types.SimpleNamespace(understood=understood, request=object() if understood else None)
        module("core.actions.planner", plan_action=plan)
        self.router = load("core.request_router", "core/request_router.py")
        self.engine_module = load("core.engine", "core/engine.py")
        self.planner = FakePlanner(decision("complete"))
        self.tools = FakeTools()
        self.engine_module.AgentSessions = lambda directory: L.Sessions.AgentSessions(
            self.tmp.name, lambda: L.AgentLoop(self.planner, self.tools))
        self.engine = self.engine_module.StellaEngine()

    def test_normal_chat_keeps_rag_sources_tools_and_memory(self):
        result = self.engine.chat("What do my documents say?", "chat")
        self.assertEqual(result["route"], "chat")
        self.assertEqual(result["sources"], [{"source": "docs", "text": "evidence"}])
        self.assertEqual(result["tools_used"], ["search_docs"])
        self.assertEqual(result["search_query"], "rewritten")
        self.assertIs(self.chat_calls[0][1], self.engine._get_memory("chat"))
        self.assertEqual(self.direct_calls, [])
        self.assertEqual(self.planner.calls, 0)

    def test_simple_action_routes_direct_without_agent_planning(self):
        result = self.engine.chat("open Safari", "chat")
        self.assertEqual(result["route"], "direct")
        self.assertEqual(self.direct_calls, ["open Safari"])
        self.assertEqual(self.planner.calls, 0)

    def test_complex_action_is_routed_before_direct_dispatch(self):
        result = self.engine.chat("open Safari and then open Finder", "chat")
        self.assertEqual(result["route"], "agent")
        self.assertEqual(result["agent"]["status"], "completed")
        self.assertEqual(self.direct_calls, [])

    def test_explanation_containing_action_words_does_not_execute(self):
        result = self.engine.chat("How do I open Safari and delete a file?", "chat")
        self.assertEqual(result["route"], "chat")
        self.assertEqual(self.direct_calls, [])
        self.assertEqual(self.tools.calls, [])

    def test_quoted_conjunction_is_not_a_second_action(self):
        routing = self.router.route_request('create folder "rock and roll"')
        self.assertEqual(routing.route, "direct")

    def test_polite_direct_request_routes_without_extra_model_call(self):
        result = self.engine.chat("Can you open Safari", "chat")
        self.assertEqual(result["route"], "direct")
        self.assertEqual(self.direct_calls, ["open Safari"])

    def test_explicit_chat_override_prevents_execution(self):
        result = self.engine.chat("open Safari", "chat", mode="chat")
        self.assertEqual(result["route"], "chat")
        self.assertEqual(self.direct_calls, [])

    def test_explicit_agent_prefix(self):
        result = self.engine.chat("/agent organize my windows", "chat")
        self.assertEqual(result["route"], "agent")
        self.assertEqual(self.engine._agent_sessions.store.read("chat")[1].goal.text, "organize my windows")

    def test_parser_failure_falls_back_without_execution(self):
        with patch.object(self.router, "plan_action", side_effect=RuntimeError("parser unavailable")):
            result = self.engine.chat("open Safari", "chat")
        self.assertEqual(result["route"], "chat")
        self.assertEqual(self.direct_calls, [])

    def test_failed_direct_action_is_not_repeated_by_agent_fallback(self):
        self.direct_result.status = "FAILED"
        result = self.engine.chat("open Safari", "chat")
        self.assertEqual(result["route"], "direct")
        self.assertEqual(len(self.direct_calls), 1)
        self.assertEqual(self.planner.calls, 0)

    def test_live_natural_volume_question_routes_to_agent(self):
        for message in (
            "Could you tell me how loud my Mac is right now?",
            "How loud is my Mac?", "What's my current system volume?",
            "Please tell me which app is in front right now?",
        ):
            with self.subTest(message=message):
                self.assertEqual(self.router.route_request(message).route, "agent")

    def test_device_mentions_in_explanations_stay_chat(self):
        for message in (
            "How do I change my Mac volume?",
            "Tell me how loud my Mac should be for a meeting",
            'Explain "how loud my Mac is right now"',
            "What is system volume and how do I change it?",
            "What if my volume is muted?",
            "Could you tell me how loud my Mac is right now and delete a file?",
        ):
            with self.subTest(message=message):
                self.assertEqual(self.router.route_request(message).route, "chat")

    def test_snapshot_question_honors_chat_override(self):
        for message in ("How loud is my Mac?", "/chat How loud is my Mac?"):
            self.assertEqual(self.router.route_request(message, "chat").route, "chat")

    def pause(self):
        self.planner = FakePlanner(decision(), decision("complete"))
        self.tools = FakeTools(observation(False, status="blocked", requires_confirmation=True), observation(data=1))
        return self.engine.chat("/agent do this action", "chat")

    def reply_ids(self, result):
        agent = result["agent"]
        return {k: agent[k] for k in ("run_id", "confirmation_id", "resume_token")}

    def test_untagged_yes_returns_prompt_without_executing(self):
        first = self.pause()
        second = self.engine.chat("yes", "chat")
        self.assertEqual(second["agent"]["confirmation_id"], first["agent"]["confirmation_id"])
        self.assertEqual(len(self.tools.calls), 1)

    def test_matching_ids_resume_exact_action_and_surface_diagnostics(self):
        first = self.pause()
        result = self.engine.chat("yes", "chat", **self.reply_ids(first))
        self.assertEqual(result["agent"]["status"], "completed")
        self.assertEqual(result["agent"]["run_id"], first["agent"]["run_id"])
        self.assertEqual(result["tools_used"], ["get_system_volume"])
        self.assertEqual(result["agent"]["diagnostics"]["steps"], 1)
        self.assertEqual(len(self.tools.calls), 2)

    def test_wrong_run_or_confirmation_never_dispatches_or_falls_back(self):
        first = self.pause()
        for key in ("run_id", "confirmation_id"):
            ids = self.reply_ids(first)
            ids[key] = "wrong"
            result = self.engine.chat("yes", "chat", **ids)
            self.assertIn("agent_error", result)
        self.assertEqual(len(self.tools.calls), 1)
        self.assertEqual(self.direct_calls, [])
        self.assertEqual(self.chat_calls, [])

    def test_duplicate_tagged_yes_cannot_start_another_goal(self):
        first = self.pause()
        self.engine.chat("yes", "chat", **self.reply_ids(first))
        result = self.engine.chat("yes", "chat", **self.reply_ids(first))
        self.assertIn("agent_error", result)
        self.assertEqual(len(self.tools.calls), 2)
        self.assertEqual(self.chat_calls, [])

    def test_clarification_resumes_original_run_and_keeps_budget(self):
        ask = decision("ask_user")
        ask.message = "Which application?"
        self.planner = FakePlanner(ask, decision("complete"))
        first = self.engine.chat("/agent arrange it", "chat")
        before = first["agent"]["diagnostics"]["global_steps_used"]
        self.assertEqual(first["agent"]["status"], "waiting_for_user")
        self.assertTrue(first["agent"]["resume_token"])
        result = self.engine.chat("Safari", "chat", **self.reply_ids(first))
        self.assertEqual(result["agent"]["status"], "completed")
        self.assertEqual(result["agent"]["run_id"], first["agent"]["run_id"])
        self.assertGreater(result["agent"]["diagnostics"]["global_steps_used"], before)
        state = self.engine._agent_sessions.store.read("chat")[1]
        self.assertEqual(state.goal.text, "arrange it")
        self.assertEqual(state.conversation_context[-1]["content"], "Safari")
        history = P.AgentPlanner(tool_adapter=self.tools)._format_history(state)
        self.assertIn("Safari", history)
        self.assertIn("Which application?", history)

    def test_stale_clarification_token_keeps_pending_question(self):
        self.planner = FakePlanner(decision("ask_user"))
        first = self.engine.chat("/agent arrange it", "chat")
        ids = self.reply_ids(first)
        ids["resume_token"] = "old"
        result = self.engine.chat("Safari", "chat", **ids)
        self.assertIn("agent_error", result)
        self.assertEqual(self.planner.calls, 1)

    def test_context_snapshot_includes_memory_metadata(self):
        memory = self.engine._get_memory("chat")
        memory.messages = [{"role": "assistant", "content": "The folder is Projects", "sources": [{"source": "docs"}]}]
        self.engine.chat("/agent open that folder", "chat")
        state = self.engine._agent_sessions.store.read("chat")[1]
        self.assertEqual(state.conversation_context[0]["sources"], [{"source": "docs"}])
        self.assertIn("Projects", state.conversation_context[0]["content"])

    def test_schema_one_snapshot_migrates_additive_context_fields(self):
        state = M.AgentState(goal=M.AgentGoal("Old run"))
        encoded = json.loads(L.Persistence.dumps_state(state))
        encoded["version"] = 1
        encoded["state"]["value"].pop("conversation_context")
        encoded["state"]["value"].pop("resume_token")
        restored = L.Persistence.loads_state(json.dumps(encoded))
        self.assertEqual(restored.run_id, state.run_id)
        self.assertEqual(restored.conversation_context, [])
        self.assertIsNone(restored.resume_token)

    def test_previous_confirmation_cannot_approve_next_action(self):
        self.planner = FakePlanner(decision(arguments={"n": 1}), decision(arguments={"n": 2}), decision("complete"))
        gate = lambda: observation(False, status="blocked", requires_confirmation=True)
        self.tools = FakeTools(gate(), observation(data=1), gate(), observation(data=2))
        first = self.engine.chat("/agent do two actions", "chat")
        second = self.engine.chat("yes", "chat", **self.reply_ids(first))
        stale = self.engine.chat("yes", "chat", **self.reply_ids(first))
        self.assertIn("agent_error", stale)
        self.assertEqual(len(self.tools.calls), 3)
        done = self.engine.chat("yes", "chat", **self.reply_ids(second))
        self.assertEqual(done["agent"]["status"], "completed")
        self.assertEqual(len(self.tools.calls), 4)

    def test_engine_restart_resumes_using_received_ids(self):
        first = self.pause()
        self.engine = self.engine_module.StellaEngine()
        result = self.engine.chat("yes", "chat", **self.reply_ids(first))
        self.assertEqual(result["agent"]["status"], "completed")
        self.assertEqual(result["agent"]["run_id"], first["agent"]["run_id"])
        self.assertEqual(len(self.tools.calls), 2)

    def test_clarification_cancellation_does_not_replan(self):
        self.planner = FakePlanner(decision("ask_user"))
        first = self.engine.chat("/agent arrange it", "chat")
        result = self.engine.chat("cancel", "chat", **self.reply_ids(first))
        self.assertEqual(result["agent"]["status"], "cancelled")
        self.assertEqual(self.planner.calls, 1)

    def test_clarification_does_not_reset_exhausted_global_budget(self):
        self.planner = FakePlanner(decision("ask_user"))
        first = self.engine.chat("/agent arrange it", "chat")
        state = self.engine._agent_sessions.store.read("chat")[1]
        state.max_global_steps = state.global_steps_used
        self.engine._agent_sessions.store.save("chat", state, "idle")
        result = self.engine.chat("Safari", "chat", **self.reply_ids(first))
        self.assertEqual(result["agent"]["reason"], "global_step_budget")
        self.assertEqual(self.planner.calls, 1)

    def test_current_schema_missing_fields_is_rejected(self):
        state = M.AgentState(goal=M.AgentGoal("Test"))
        payload = json.loads(L.Persistence.dumps_state(state))
        payload["state"]["value"].pop("conversation_context")
        with self.assertRaises(L.Persistence.AgentStorageError):
            L.Persistence.loads_state(json.dumps(payload))

    def test_api_models_retain_agent_metadata_and_forward_ids(self):
        # Real pydantic models and endpoint function; fake web server decorator.
        class App:
            def __init__(self, **kw): pass
            def get(self, *args, **kw): return lambda f: f
            def post(self, *args, **kw): return lambda f: f
        fake_fastapi = types.ModuleType("fastapi")
        fake_fastapi.FastAPI = App
        fake_fastapi.HTTPException = RuntimeError
        sys.modules["fastapi"] = fake_fastapi
        api = load("integration_api", "api.py")
        first = self.pause()
        api.stella = self.engine
        request = api.ChatRequest(message="yes", conversation_id="chat", **self.reply_ids(first))
        response = api.ChatResponse(**api.chat(request))
        self.assertEqual(response.agent["status"], "completed")
        self.assertEqual(response.agent["run_id"], first["agent"]["run_id"])
        with self.assertRaises(ValueError):
            api.ChatRequest(message="x", conversation_id="chat", mode="bad")


if __name__ == "__main__":
    unittest.main()