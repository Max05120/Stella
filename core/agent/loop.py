"""Agent runtime with explicit terminal outcomes (Phase 6F.2).

Budgets count planning, model/repair attempts and tool dispatches across resume.
Deadlines are cooperative checks: in-flight synchronous calls are not killed.
"""

from __future__ import annotations

from copy import deepcopy
from uuid import uuid4
import json

from core.agent.models import (
    AgentDecision,
    AgentDecisionType,
    AgentObservation,
    AgentState,
    AgentStatus,
    AgentStep,
    AgentTerminationReason,
)
from core.agent.budgets import (
    AgentBudgetExceeded, claim_operation, check_runtime, freeze_runtime,
)
from core.actions.confirmation import confirmation_intent
from core.agent.confirmations import (
    create_confirmation, record_matches, confirmation_expired,
    audit_confirmation, close_confirmation,
)
from core.agent.planner import AgentPlanner
from core.agent.progress import check_action, check_recovery_action, record_dispatch, record_result
from core.agent.recovery import annotate_failure, failure_explanation
from core.agent.tool_adapter import AgentToolAdapter


class AgentLoopError(Exception):
    """The caller requested an invalid runtime transition."""


class AgentLoop:
    def __init__(
        self,
        planner: AgentPlanner | None = None,
        tools: AgentToolAdapter | None = None,
    ):
        self.checkpoint = lambda state: None
        self.tools = tools if tools is not None else AgentToolAdapter()
        self.planner = (
            planner if planner is not None
            else AgentPlanner(tool_adapter=self.tools)
        )

    def _finish(self, state, status, reason, message) -> AgentState:
        # Never overwrite a run that already finished.
        if state.is_terminal:
            return state

        close_confirmation(state, reason)
        freeze_runtime(state)
        state.final_observation = deepcopy(state.latest_observation)
        state.status = status
        state.termination_reason = reason
        state.pending_confirmation = None
        state.pending_confirmation_step = None
        state.resume_token = None

        latest = state.latest_observation
        if (status in {AgentStatus.FAILED, AgentStatus.BLOCKED, AgentStatus.LIMIT_EXCEEDED}
                and latest is not None and not latest.success
                and not (latest.status == "blocked" and latest.requires_confirmation)):
            message += " " + failure_explanation(latest)

        successful = sum(
            bool(step.observation and step.observation.success)
            for step in state.steps
        )
        if status != AgentStatus.COMPLETED and successful:
            message += (
                f" {successful} earlier tool step(s) succeeded; "
                "those actions have not been undone."
            )
        readbacks = state.inspection_results()
        if readbacks:
            message += "\n\nObserved results:\n" + "\n".join(
                f"- Step {row['step_number']}: {row['summary']}"
                for row in readbacks
            )
        state.final_answer = message

        print(
            "[AGENT TERMINATION] "
            + json.dumps(state.termination_diagnostics(), ensure_ascii=False)
        )
        return state

    def _budget_stop(self, state, exc):
        return self._finish(state, AgentStatus.LIMIT_EXCEEDED, exc.reason, str(exc))

    def _runtime_expired(self, state):
        try:
            check_runtime(state)
        except AgentBudgetExceeded as exc:
            self._budget_stop(state, exc)
            return True
        return False

    def _step_limit(self, state):
        return self._finish(
            state, AgentStatus.LIMIT_EXCEEDED, AgentTerminationReason.MAX_STEPS,
            "I reached the tool-step limit and could not verify the whole goal "
            "as complete. No additional action was executed.",
        )

    def _stop_if_exhausted(self, state: AgentState, *, pending_slot=False) -> bool:
        if self._runtime_expired(state):
            return True
        if state.failure_count >= state.max_failures:
            self._finish(
                state, AgentStatus.LIMIT_EXCEEDED,
                AgentTerminationReason.MAX_FAILURES,
                "I stopped because this task reached its failure limit.",
            )
            return True
        if state.consecutive_failures >= state.max_consecutive_failures:
            self._finish(
                state, AgentStatus.LIMIT_EXCEEDED,
                AgentTerminationReason.MAX_CONSECUTIVE_FAILURES,
                "I stopped after too many consecutive tool failures.",
            )
            return True
        if state.no_progress_steps >= state.max_no_progress_steps:
            self._finish(
                state, AgentStatus.BLOCKED,
                AgentTerminationReason.NO_PROGRESS,
                "I stopped because recent tool steps produced no new evidence "
                "or successful new action.",
            )
            return True
        if state.step_count > state.max_steps:
            self._step_limit(state)
            return True
        if state.step_count == state.max_steps and not pending_slot:
            last = state.latest_step
            if (state.completion_check_used or last is None
                    or last.observation is None or not last.observation.success):
                self._step_limit(state)
                return True
            # Allow one budgeted completion-only decision after the last
            # successful tool. Runtime dispatch still refuses all new tools.
        return False

    def _execute(
        self,
        decision: AgentDecision,
        *,
        confirmed: bool = False,
    ) -> AgentObservation:
        try:
            observation = self.tools.execute(
                decision.capability,
                deepcopy(decision.arguments),
                confirmed=confirmed,
            )
            if not isinstance(observation, AgentObservation):
                raise TypeError("Tool adapter did not return AgentObservation")
            observation = deepcopy(observation)
            observation.capability = decision.capability
            return annotate_failure(observation)
        except Exception as exc:
            # Preserve an unexpected adapter exception as an observation.
            # Do not retry a possibly partially applied action here.
            observation = AgentObservation(
                capability=decision.capability or "unknown",
                success=False,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                message="The capability did not return a normal result.",
            )
            return annotate_failure(observation, exception=exc)

    def _run_tool_step(self, state, decision) -> None:
        if not decision.capability:
            state.failure_count += 1
            self._finish(
                state, AgentStatus.FAILED,
                AgentTerminationReason.INVALID_DECISION,
                "The planner selected a tool without naming a capability.",
            )
            return

        violation = check_action(state, decision)
        if violation is not None:
            self._finish(
                state, AgentStatus.BLOCKED, violation.reason, violation.message,
            )
            return

        if state.step_count >= state.max_steps:
            self._step_limit(state)
            return
        claim_operation(state, "tool")
        record_dispatch(state, decision)
        step = AgentStep(
            number=state.step_count + 1,
            decision=deepcopy(decision),
        )
        state.steps.append(step)
        self.checkpoint(state)
        observation = self._execute(step.decision)
        step.observation = observation
        self.checkpoint(state)

        if (
            observation.requires_confirmation
            and not observation.success
            and observation.status == "blocked"
        ):
            check_runtime(state)
            state.pending_confirmation = deepcopy(step.decision)
            state.pending_confirmation_step = step.number
            record = create_confirmation(state, step.decision, step.number)
            state.status = AgentStatus.WAITING_FOR_CONFIRMATION
            state.final_answer = record.explanation
            return

        record_result(state, step.decision, observation)
        self.checkpoint(state)
        check_runtime(state)
        # Recovery still sees the failed observation on the next iteration.

    def run(self, state: AgentState) -> AgentState:
        if state.is_terminal or state.status in {
            AgentStatus.WAITING_FOR_CONFIRMATION,
            AgentStatus.WAITING_FOR_USER,
        }:
            return state

        state.status = AgentStatus.RUNNING
        state.final_answer = None

        while not self._stop_if_exhausted(state):
            completion_only = state.step_count == state.max_steps
            try:
                claim_operation(state, "decision")
                if completion_only:
                    state.completion_check_used = True
                decision = self.planner.decide(state)
                check_runtime(state)
            except AgentBudgetExceeded as exc:
                return self._budget_stop(state, exc)
            except Exception as exc:
                if self._runtime_expired(state):
                    return state
                reason = getattr(exc, "termination_reason", None)
                if reason == AgentTerminationReason.MAX_STEPS:
                    return self._step_limit(state)
                if reason in {
                    AgentTerminationReason.REPEATED_DECISION,
                    AgentTerminationReason.REPEATED_FAILED_STRATEGY,
                    AgentTerminationReason.NO_SAFE_ACTION,
                }:
                    return self._finish(
                        state, AgentStatus.BLOCKED, reason,
                        "The planner could not produce an allowed recovery strategy. " + str(exc),
                    )
                state.failure_count += 1
                return self._finish(
                    state, AgentStatus.FAILED,
                    AgentTerminationReason.PLANNER_ERROR,
                    "I stopped because the planner could not produce a valid "
                    f"next step: {type(exc).__name__}: {exc}",
                )

            if not isinstance(decision, AgentDecision):
                state.failure_count += 1
                return self._finish(
                    state, AgentStatus.FAILED,
                    AgentTerminationReason.INVALID_DECISION,
                    "The planner returned an invalid decision object.",
                )

            kind = decision.decision_type
            if completion_only and kind != AgentDecisionType.COMPLETE:
                return self._step_limit(state)
            if kind == AgentDecisionType.TOOL:
                try:
                    self._run_tool_step(state, decision)
                except AgentBudgetExceeded as exc:
                    return self._budget_stop(state, exc)
                if state.status != AgentStatus.RUNNING:
                    return state
                continue

            if kind == AgentDecisionType.COMPLETE:
                # Keep a runtime backstop even if a custom planner skips
                # AgentPlanner's existing completion validation.
                latest = state.latest_step
                if latest is not None and (
                    latest.observation is None or not latest.observation.success
                ):
                    return self._finish(
                        state, AgentStatus.FAILED,
                        AgentTerminationReason.INVALID_COMPLETION,
                        "I could not confirm completion because the latest "
                        "tool step did not succeed.",
                    )
                return self._finish(
                    state, AgentStatus.COMPLETED,
                    AgentTerminationReason.GOAL_SATISFIED,
                    decision.message or "The requested goal has been completed.",
                )

            if kind in {AgentDecisionType.ASK_USER, AgentDecisionType.RESPOND}:
                state.resume_token = str(uuid4())
                state.status = AgentStatus.WAITING_FOR_USER
                state.final_answer = (
                    decision.message or "I need more information to continue."
                )
                return state

            if kind == AgentDecisionType.ABORT:
                observation = state.latest_observation
                blocked = (
                    observation is None
                    or observation.success
                    or observation.status == "blocked"
                    or observation.recovery_type == "permission_denied"
                )
                return self._finish(
                    state,
                    AgentStatus.BLOCKED if blocked else AgentStatus.FAILED,
                    (
                        AgentTerminationReason.NO_SAFE_ACTION if blocked
                        else AgentTerminationReason.UNRECOVERABLE_FAILURE
                    ),
                    decision.message or "I could not find a safe way to finish this task.",
                )

            state.failure_count += 1
            return self._finish(
                state, AgentStatus.FAILED,
                AgentTerminationReason.INVALID_DECISION,
                "The planner returned an unsupported decision type.",
            )

        return state

    def _check_confirmation_id(self, state, confirmation_id):
        if confirmation_id is None:
            return  # Compatibility with existing single-conversation callers.
        if state.confirmation is None or state.confirmation.confirmation_id != confirmation_id:
            # A stale tagged response must not consume the current request.
            if state.confirmation is not None:
                audit_confirmation(state, "stale_response", state.confirmation)
            raise AgentLoopError("This reply does not match the current confirmation.")

    def reply_to_confirmation(self, state, message, *, confirmation_id=None):
        if state.status != AgentStatus.WAITING_FOR_CONFIRMATION:
            raise AgentLoopError("Agent is not waiting for confirmation.")
        self._check_confirmation_id(state, confirmation_id)
        intent = confirmation_intent(message)
        if intent is None:
            raise AgentLoopError("Use an explicit yes, no, or cancel response.")
        normalized = " ".join(message.lower().strip().split()).rstrip(".!?")
        if normalized in {"cancel", "stop", "never mind", "nevermind"}:
            return self.cancel(state, confirmation_id=confirmation_id)
        return self.confirm(state, approved=intent, confirmation_id=confirmation_id)

    def invalidate_confirmation(self, state):
        if state.status != AgentStatus.WAITING_FOR_CONFIRMATION:
            raise AgentLoopError("Agent is not waiting for confirmation.")
        return self._finish(
            state, AgentStatus.CANCELLED,
            AgentTerminationReason.CONFIRMATION_SUPERSEDED,
            "I cancelled the pending approval because the conversation moved on.",
        )

    def confirm(self, state: AgentState, approved: bool, *, confirmation_id=None) -> AgentState:
        if state.status != AgentStatus.WAITING_FOR_CONFIRMATION:
            raise AgentLoopError("Agent is not waiting for confirmation.")
        if not isinstance(approved, bool):
            raise AgentLoopError("approved must be True or False.")
        self._check_confirmation_id(state, confirmation_id)

        if not approved:
            return self._finish(
                state, AgentStatus.CANCELLED,
                AgentTerminationReason.CONFIRMATION_REJECTED,
                "The pending action was not confirmed, so I stopped the task.",
            )

        decision = state.pending_confirmation
        step_number = state.pending_confirmation_step
        matches = [step for step in state.steps if step.number == step_number]

        # Validate all saved state BEFORE executing the pending capability.
        if (
            decision is None
            or decision.decision_type != AgentDecisionType.TOOL
            or not decision.capability
            or not record_matches(state, decision, step_number)
            or len(matches) != 1
            or matches[0].decision != decision
            or matches[0].observation is None
            or matches[0].observation.success
            or matches[0].observation.status != "blocked"
            or not matches[0].observation.requires_confirmation
        ):
            return self._finish(
                state, AgentStatus.FAILED,
                AgentTerminationReason.INVALID_CONFIRMATION,
                "The saved confirmation is inconsistent. No pending action was executed.",
            )

        # Approval owns its existing tool-step slot, but does not reset any
        # cumulative, consecutive, or no-progress counter.
        if self._stop_if_exhausted(state, pending_slot=True):
            return state
        recovery_violation = check_recovery_action(state, decision)
        if recovery_violation is not None:
            return self._finish(state, AgentStatus.BLOCKED,
                                recovery_violation.reason, recovery_violation.message)
        if confirmation_expired(state):
            return self._finish(
                state, AgentStatus.CANCELLED,
                AgentTerminationReason.CONFIRMATION_EXPIRED,
                "That approval expired, so I stopped the task without executing "
                "the pending action. Start a new task if you still want it done.",
            )

        try:
            claim_operation(state, "tool")
        except AgentBudgetExceeded as exc:
            return self._budget_stop(state, exc)

        pending_step = matches[0]
        record = state.confirmation
        audit_confirmation(state, "approved", record)
        # Consume confirmation before execution. A second sequential confirm
        # cannot dispatch this saved action again.
        state.pending_confirmation = None
        state.pending_confirmation_step = None
        state.confirmation = None
        state.status = AgentStatus.RUNNING
        state.final_answer = None

        self.checkpoint(state)
        observation = self._execute(pending_step.decision, confirmed=True)
        pending_step.observation = observation
        audit_confirmation(
            state, "execution_result", record,
            success=observation.success, status=observation.status,
        )
        record_result(state, pending_step.decision, observation)
        self.checkpoint(state)
        if self._runtime_expired(state):
            return state

        if not observation.success:
            if observation.status == "blocked" and observation.requires_confirmation:
                return self._finish(
                    state, AgentStatus.BLOCKED,
                    AgentTerminationReason.CONFIRMATION_STILL_BLOCKED,
                    "The action remained blocked after approval. I stopped instead "
                    "of requesting the same approval again.",
                )

        return self.run(state)

    def cancel(self, state: AgentState, *, confirmation_id=None) -> AgentState:
        """Cancel a ready or paused run; active-call cancellation is later work."""
        if state.is_terminal:
            return state
        self._check_confirmation_id(state, confirmation_id)
        if state.status == AgentStatus.RUNNING:
            raise AgentLoopError(
                "Active-call cancellation is not implemented in Phase 6F.2."
            )
        return self._finish(
            state, AgentStatus.CANCELLED,
            AgentTerminationReason.USER_CANCELLED,
            "I cancelled the task.",
        )