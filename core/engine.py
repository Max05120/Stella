"""
engine.py

Main application interface for Stella.
"""

from collections import OrderedDict
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
        self._agent_loop = AgentLoop()
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

        # Do not overwrite an existing paused agent.
        existing = (
            self._active_agent_states.get(conversation_id)
        )

        if existing is not None:
            answer = (
                "An agent task is already waiting " \
                "for your response."
            )
            return {
                "answer": answer,
                "sources": [],
                "search_query": message,
                "tools_used": self._active_agent_states(existing),
            }
        
        state = AgentState(
            goal=AgentGoal(text=message)
        )

        state = self._agent_loop.run(state)

        return self._agent_response(
            state=state,
            conversation_id=conversation_id,
            memory=memory,
            user_message=message,
        )
    
    def chat(
        self,
        message: str,
        conversation_id: str,
    ):
        memory = self._get_memory(
            conversation_id
        )

        #------------------------------------------------------
        # -1. Resume a paused autonomous agent confirmation
        #------------------------------------------------------

        agent_state = self._active_agent_states.get(conversation_id)

        if (
            agent_state is not None
            and agent_state.status
            == AgentStatus.WAITING_FOR_CONFIRMATION
        ):
            intent = confirmation_intent(message)

            #------------------------------------------------------
            # User approved agent action
            #------------------------------------------------------

            if intent is True:

                state = self._agent_loop.confirm(
                    agent_state,
                    approved=True,
                )

                return self._agent_response(
                    state=state,
                    conversation_id=conversation_id,
                    memory=memory,
                    user_message=message,
                )
            
            #------------------------------------------------------
            # User denied agent action.
            #------------------------------------------------------

            if intent is False:
                state = self._agent_loop.confirm(
                    agent_state,
                    approved=False,
                )

                return self._agent_response(
                    state=state,
                    conversation_id=conversation_id,
                    memory=memory,
                    user_message=message,
                )
            # -----------------------------------------------------
            # Unrelated message.
            #
            # Invalidate the pending destructive agent action
            # so a later unrelated "yes" cannot execute it.
            # -----------------------------------------------------

            self._agent_loop.confirm(
                agent_state,
                approved=False,
            )

            self._active_agent_states.pop(
                conversation_id,
                None,
            )

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

        action_result = run_action(message)

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

        # -----------------------------------------------------
        # 2. Otherwise continue through Stella's normal
        #    conversational / RAG / tool pipeline
        # -----------------------------------------------------

        (
            answer,
            results,
            search_query,
            tools_used,
            timings,
        ) = ask(
            message,
            memory,
        )

        return {
            "answer": answer,
            "sources": results,
            "search_query": search_query,
            "tools_used": tools_used,
        }
    
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

    while True:

        message = input("You: ").strip()

        if message.lower() == "exit":
            break

        response = stella.chat(
            message,
            conversation_id,
        )

        print(
            "\nStella:",
            response["answer"],
            "\n",
        )