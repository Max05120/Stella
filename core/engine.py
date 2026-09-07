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
        self._pending_actions = {}

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

    def chat(
        self,
        message: str,
        conversation_id: str,
    ):
        memory = self._get_memory(
            conversation_id
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