"""
engine.py

Main application interface for Stella.
"""

from collections import OrderedDict
from pathlib import Path
from core.request_router import route_request
from core.agent.sessions import AgentSessions
from core.agent.persistence import AgentStorageError
from core.agent.loop import AgentLoopError
from core.actions.responses import (
    format_action_success,
    format_action_failure,
    format_action_confirmation,
    format_action_cancelled,
)
from core.memory import ConversationMemory
from core.actions.runner import (
    run_action,
    confirm_action,
)
from core.agent import (
    AgentGoal,
    AgentLoop,
    AgentState,
    AgentStatus,
)
from core.actions.confirmation import (
    confirmation_intent,
)
from core.rag import ask
import time

class StellaEngine:

    MAX_MEMORY_CACHE = 32

    def __init__(self):
        self._memories: OrderedDict[
            str,
            ConversationMemory
        ] = OrderedDict()
        # Deterministic action confirmation
        self._pending_actions = {}
        # Agent runtime.
        self._agent_sessions = AgentSessions(
            Path(__file__).resolve().parents[1] / ".stella" / "agents"
        )
        # Paused agent runs, keyed by conversation.
        self._active_agent_states: dict[
            str,
            AgentState,
        ] = {}

    def _get_memory(
        self,
        conversation_id: str,
    ) -> ConversationMemory:

        memory = self._memories.get(conversation_id)

        if memory is not None:
            self._memories.move_to_end(
                conversation_id
            )
            return memory

        memory = ConversationMemory(
            conversation_id
        )

        self._memories[
            conversation_id
        ] = memory

        # Prevent unbounded cache growth.
        if (
            len(self._memories)
            > self.MAX_MEMORY_CACHE
        ):
            self._memories.popitem(
                last=False
            )

        return memory
    
    def _record_action_turn(
        self,
        memory: ConversationMemory,
        user_message: str,
        assistant_message: str,
        capability: str | None = None,
    ):
        memory.add_user_message(
            user_message
        )

        memory.add_assistant_message(
            assistant_message,
            sources=[],
            tools_used=(
                [capability]
                if capability
                else []
            ),
        )
    def _agent_tools_used(
            self,
            state: AgentState,
    ) -> list[str]:
        """
        Return the unique capabilities used during 
        the current agent run, preserving order.
        """

        tools_used = []

        for step in state.steps:

            capability = step.decision.capability
            if capability is None:
                continue

            if capability not in tools_used:
                tools_used.append(
                    capability
                )
        return tools_used
    
    def _record_agent_turn(
            self,
            memory: ConversationMemory,
            user_message: str,
            assistant_message: str,
            state: AgentState,
    ):
        """
        Record one conversational turn involving
        an autonomous agent run.
        """

        memory.add_user_message(user_message)
        memory.add_assistant_message(
            assistant_message,
            sources=[],
            tools_used=self._agent_tools_used(
                state
            ),
        )

    def _agent_response(
            self,
            state: AgentState,
            conversation_id: str,
            memory: ConversationMemory,
            user_message: str,
    ):
        """
        Store or clear agent state depending on it's 
        status, record the turn, and build the normal
        Stella response dicitionary.
        """

        waiting = {
            AgentStatus.WAITING_FOR_CONFIRMATION,
            AgentStatus.WAITING_FOR_USER,
        }

        if state.status in waiting:
            self._active_agent_states[conversation_id] = state
        
        else:
            self._active_agent_states.pop(conversation_id, None)

        answer = (
            state.final_answer or
            "The agent run finished."
        )

        tools_used =(
            self._agent_tools_used(state)
        )

        self._record_agent_turn(
            memory=memory,
            user_message=user_message,
            assistant_message=answer,
            state=state,
        )

        return {
            "answer": answer,
            "sources": [],
            "search_query": user_message,
            "tools_used": tools_used,
            "route": "agent",
            "agent": {
                "run_id": state.run_id,
                "status": state.status.value,
                "reason": state.termination_reason.value if state.termination_reason else None,
                "confirmation_id": state.confirmation.confirmation_id if state.confirmation else None,
                "resume_token": state.resume_token,
                "diagnostics": state.termination_diagnostics(),
            },
        }

    def _agent_session_error(self, message, error):
        return {
            "answer": str(error),
            "sources": [],
            "search_query": message,
            "tools_used": [],
            "route": "agent",
            "agent_error": str(error),
        }

    def run_agent_goal(
            self,
            message: str,
            conversation_id: str,
    ):
        """
        Start a new autonomous agent goal.

        Phase 6H will later decide automatically 
        when normal chat should route here.
        """

        memory = self._get_memory(conversation_id)

        try:
            context = memory.get_messages(limit=12, include_metadata=True)
            context = [dict(item, content=str(item.get("content", ""))[:1000]) for item in context]
            state = self._agent_sessions.start(conversation_id, message, context=context)
        except (AgentStorageError, AgentLoopError, OSError) as exc:
            return self._agent_session_error(message, exc)

        return self._agent_response(
            state=state,
            conversation_id=conversation_id,
            memory=memory,
            user_message=message,
        )
    
    def chat(self, message, conversation_id, *, mode="auto", run_id=None,
             confirmation_id=None, resume_token=None):
        started = time.perf_counter()
        try:
            # Serialize routing as well as execution within a conversation.
            with self._agent_sessions.store.locked("engine-request:" + conversation_id):
                result = self._chat(message, conversation_id, mode=mode,
                                    run_id=run_id, confirmation_id=confirmation_id,
                                    resume_token=resume_token)
                result.setdefault("route", "direct")
                return result
        except (AgentStorageError, AgentLoopError, OSError) as exc:
            return self._agent_session_error(message, exc)
        finally:
            print(f"[STELLA REQUEST] total_ms={(time.perf_counter()-started)*1000:.1f}")

    def _chat(
        self,
        message: str,
        conversation_id: str,
        *, mode="auto", run_id=None, confirmation_id=None, resume_token=None,
    ):
        memory = self._get_memory(
            conversation_id
        )

        #------------------------------------------------------
        # -1. Resume a paused autonomous agent confirmation
        #------------------------------------------------------

        try:
            agent_state = self._agent_sessions.reply(
                conversation_id, message, run_id=run_id,
                confirmation_id=confirmation_id, resume_token=resume_token,
                require_ids=True,
            )
        except (AgentStorageError, AgentLoopError, OSError) as exc:
            return self._agent_session_error(message, exc)

        if agent_state is not None:
            return self._agent_response(
                state=agent_state,
                conversation_id=conversation_id,
                memory=memory,
                user_message=message,
            )
        self._active_agent_states.pop(conversation_id, None)

        # -----------------------------------------------------
        # 0. Resolve a pending destructive-action confirmation
        # -----------------------------------------------------

        pending = self._pending_actions.get(
            conversation_id
        )

        if pending is not None:
            intent = confirmation_intent(
                message
            )

            if intent is True:
                self._pending_actions.pop(
                    conversation_id,
                    None,
                )

                action_result = confirm_action(
                    pending
                )

                if action_result.status == "SUCCESS":
                    answer = format_action_success(
                        action_result
                    )
                else:
                    answer = format_action_failure(
                        action_result
                    )

                capability = (
                    action_result.capability
                )

                self._record_action_turn(
                    memory=memory,
                    user_message=message,
                    assistant_message=answer,
                    capability=capability,
                )

                return {
                    "answer": answer,
                    "sources": [],
                    "search_query": message,
                    "tools_used": (
                        [capability]
                        if capability
                        else []
                    ),
                }

            if intent is False:
                self._pending_actions.pop(
                    conversation_id,
                    None,
                )

                answer = format_action_cancelled(
                    pending
                )

                self._record_action_turn(
                    memory=memory,
                    user_message=message,
                    assistant_message=answer,
                    capability=None,
                )

                return {
                    "answer": answer,
                    "sources": [],
                    "search_query": message,
                    "tools_used": [],
                }

            # Any unrelated message invalidates the pending action.
            #
            # This prevents a stale destructive request from being
            # confirmed several turns later by an unrelated "yes".
            self._pending_actions.pop(
                conversation_id,
                None,
            )

        # -----------------------------------------------------
        # 1. Try Stella's deterministic Mac action system
        # -----------------------------------------------------

        routing = route_request(message, mode)
        if routing.route == "agent":
            return self.run_agent_goal(routing.message, conversation_id)
        if routing.route == "chat":
            return self._conversation_response(
                routing.message, memory,
                "Routing selected conversation. No Mac capability was executed. "
                "Answer using the normal memory/RAG pipeline; ask for clarification "
                "if an action was intended. Do not claim a Mac action succeeded.",
            )

        action_started = time.perf_counter()

        action_result = run_action(routing.message)

        action_elapsed = (
            time.perf_counter() - action_started
        ) * 1000

        print(
            "[STELLA ACTION] "
            f"status={action_result.status} "
            f"elapsed_ms={action_elapsed:.1f}"
        )

        if action_result.status == "SUCCESS":
            answer = format_action_success(
            action_result
        )
            capability = action_result.capability

            self._record_action_turn(
                memory=memory,
                user_message=message,
                assistant_message=answer,
                capability=capability,
            )

            return {
                "answer": answer,
                "sources": [],
                "search_query": message,
                "tools_used": (
                    [capability]
                    if capability
                    else []
                ),
            }


        if action_result.status == "FAILED":

            answer = format_action_failure(
            action_result
            )

            capability = action_result.capability

            self._record_action_turn(
                memory=memory,
                user_message=message,
                assistant_message=answer,
                capability=capability,
            )


            return {
                "answer": answer,
                "sources": [],
                "search_query": message,
                "tools_used": (
                    [capability]
                    if capability
                    else []
                ),
            }


        if action_result.status == "BLOCKED":
            if action_result.requires_confirmation:
                self._pending_actions[
                    conversation_id
                ] = action_result

                answer = (
                    format_action_confirmation(
                        action_result
                    )
                )

                self._record_action_turn(
                    memory=memory,
                    user_message=message,
                    assistant_message=answer,
                    capability=action_result.capability,
                )

                return {
                    "answer": answer,
                    "sources": [],
                    "search_query": message,
                    "tools_used": [],
                }

            answer = action_result.message

            self._record_action_turn(
                memory=memory,
                user_message=message,
                assistant_message=answer,
                capability=action_result.capability,
            )

            return {
                "answer": answer,
                "sources": [],
                "search_query": message,
                "tools_used": [],
            }
        
        if action_result.status in {
            "IMPLEMENTABLE",
            "UNKNOWN",
        }:
            answer = action_result.message

            gap = getattr(
                action_result.decision,
                "gap",
                None,
            )

            if (
                gap is not None
                and getattr(
                    gap,
                    "explanation",
                    None,
                )
            ):
                answer += " " + gap.explanation

            self._record_action_turn(
                memory,
                message,
                answer,
            )

            return {
                "answer": answer,
                "sources": [],
                "search_query": message,
                "tools_used": [],
            }

        action_context = None

        if (
            action_result.status
            == "ACTION_NOT_UNDERSTOOD"
        ):
            action_context = (
                "The deterministic Mac action parser could not resolve "
                "an action from this request. No Mac action was executed. "
                "If the user wants a Mac action, ask for the missing "
                "target or explain the limitation. "
                "If this is an information request, answer normally "
                "using available evidence. "
                "Do not claim any Mac action succeeded."
            )
        # -----------------------------------------------------
        # 2. Otherwise continue through Stella's normal
        #    conversational / RAG / tool pipeline
        # -----------------------------------------------------

        return self._conversation_response(message, memory, action_context)

    def _conversation_response(self, message, memory, action_context=None):
        answer, results, search_query, tools_used, timings = ask(
            message, memory, action_context=action_context,
        )
        return {"answer": answer, "sources": results,
                "search_query": search_query, "tools_used": tools_used,
                "route": "chat"}

    def clear_memory(
        self,
        conversation_id: str,
    ):
        memory = self._get_memory(
            conversation_id
        )

        memory.clear()

    def get_history(
        self,
        conversation_id: str,
    ):
        return self._get_memory(
            conversation_id
        ).get_full_history()

    def list_conversations(self):
        return ConversationMemory.list_all()

    def create_conversation(self) -> str:
        conversation_id = (
            ConversationMemory.create_new()
        )

        # Cache it immediately.
        self._memories[
            conversation_id
        ] = ConversationMemory(
            conversation_id
        )

        return conversation_id


if __name__ == "__main__":

    stella = StellaEngine()

    conversation_id = (
        stella.create_conversation()
    )

    print("STELLA\n======\n")

    pending_ids = {}
    while True:

        message = input("You: ").strip()

        if message.lower() == "exit":
            break

        response = stella.chat(
            message,
            conversation_id,
            **pending_ids,
        )
        agent = response.get("agent") or {}
        pending_ids = (
            {key: agent.get(key) for key in ("run_id", "confirmation_id", "resume_token")}
            if agent.get("status") in {"waiting_for_confirmation", "waiting_for_user"}
            else {}
        )

        print(
            "\nStella:",
            response["answer"],
            "\n",
        )