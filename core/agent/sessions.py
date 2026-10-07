"""Durable entry point. All production agent starts/resumes use this service."""
from uuid import uuid4
from core.actions.confirmation import confirmation_intent
from core.agent.models import AgentGoal, AgentState, AgentStatus, AgentTerminationReason
from core.agent.loop import AgentLoop, AgentLoopError
from core.agent.persistence import AgentStore


class AgentSessions:
    def __init__(self, directory, loop_factory=AgentLoop):
        self.store = AgentStore(directory)
        self.loop_factory = loop_factory

    def _loop(self, conversation):
        loop = self.loop_factory()
        loop.checkpoint = lambda state: self.store.save(conversation, state, "busy")
        return loop

    def _load(self, conversation, loop):
        saved = self.store.read(conversation)
        if saved is None:
            return None, False
        phase, state = saved
        if phase == "busy" and not state.is_terminal:
            # We own the OS lock: an earlier owner exited without committing.
            # It may have dispatched an external action. Never replay it.
            loop._finish(state, AgentStatus.BLOCKED,
                         AgentTerminationReason.INTERRUPTED_RUN,
                         "The previous agent request was interrupted. An action may "
                         "have run without a saved result. Check its outcome before "
                         "starting a new task; I have not repeated it.")
            self.store.save(conversation, state, "idle")
            return state, True
        return state, False

    def start(self, conversation, goal, *, context=None):
        with self.store.locked(conversation):
            loop = self._loop(conversation)
            existing, recovered = self._load(conversation, loop)
            if recovered or (existing is not None and not existing.is_terminal):
                return existing
            state = AgentState(goal=AgentGoal(goal), conversation_context=list(context or []))
            self.store.save(conversation, state, "busy")
            state = loop.run(state)
            self.store.save(conversation, state, "idle")
            return state

    def reply(self, conversation, message, *, confirmation_id=None,
              run_id=None, resume_token=None, require_ids=False):
        """Return state if handled, None to continue normal action/chat routing."""
        with self.store.locked(conversation):
            loop = self._loop(conversation)
            state, recovered = self._load(conversation, loop)
            if recovered:
                return state
            if state is None or state.is_terminal:
                if any((run_id, confirmation_id, resume_token)):
                    raise AgentLoopError("That agent reply is stale; no action was executed.")
                return None
            if run_id is not None and run_id != state.run_id:
                raise AgentLoopError("That reply belongs to another agent run.")
            if state.status == AgentStatus.WAITING_FOR_USER:
                if state.resume_token is None:
                    state.resume_token = str(uuid4())
                    self.store.save(conversation, state, "idle")
                if confirmation_id is not None:
                    raise AgentLoopError("This run is waiting for clarification, not approval.")
                if resume_token is not None and resume_token != state.resume_token:
                    raise AgentLoopError("That clarification reply is stale.")
                if require_ids and (run_id is None or resume_token is None):
                    return state
                if not message.strip():
                    return state
                self.store.save(conversation, state, "busy")
                if message.strip().lower().rstrip(".!?") in {"cancel", "stop", "never mind", "nevermind"}:
                    state = loop.cancel(state)
                else:
                    state.conversation_context.extend([
                        {"role": "assistant", "content": state.final_answer or "Clarification requested"},
                        {"role": "user", "content": message[:4000]},
                    ])
                    state.conversation_context = state.conversation_context[-16:]
                    state.resume_token = None
                    state.status = AgentStatus.READY
                    state = loop.run(state)
                self.store.save(conversation, state, "idle")
                return state
            if state.status != AgentStatus.WAITING_FOR_CONFIRMATION:
                return state
            if resume_token is not None:
                raise AgentLoopError("This run needs confirmation, not clarification.")
            if require_ids and confirmation_intent(message) is not None and (run_id is None or confirmation_id is None):
                return state
            # Validate a tagged reply before making the request busy.
            try:
                loop._check_confirmation_id(state, confirmation_id)
            except AgentLoopError:
                self.store.save(conversation, state, "idle")
                raise
            if confirmation_intent(message) is None:
                loop.invalidate_confirmation(state)
                self.store.save(conversation, state, "idle")
                return None
            self.store.save(conversation, state, "busy")
            state = loop.reply_to_confirmation(state, message,
                                              confirmation_id=confirmation_id)
            self.store.save(conversation, state, "idle")
            return state