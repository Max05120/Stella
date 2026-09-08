"""
core/agent/loop.py

Autonomous Observe -> Decide -> Act runtime for Stella.

The loop repeatedly:

    1. Examines current AgentState
    2. Asks AgentPlanner for ONE next decision.
    3. Executes that decision when it is a tool.
    4. Records the resulting observation.
    5. Repeats until the goal is complete or execution must stop

The agent never calls capability handlers directly.
"""


from __future__ import annotations

from core.agent.models import (
    AgentDecision,
    AgentDecisionType,
    AgentState,
    AgentStatus,
    AgentStep,
)

from core.agent.planner import (
    AgentPlanner,
    AgentPlannerError,
)

from core.agent.tool_adapter import (
    AgentToolAdapter,
)

class AgentLoopError(Exception):
    """
    Raised when the autonomous loop itself cannot continue safely.
    """

class AgentLoop:
    """
    Stella's goal-directed autonomous execution loop
    """

    def __init__(
        self,
        planner: AgentPlanner | None = None,
        tools: AgentToolAdapter | None = None,
    ):
        self.tools = (
            tools
            or AgentToolAdapter()
        )

        self.planner = (
            planner
            or AgentPlanner(

                tool_adapter=self.tools
            )
        )

    def _run_tool_step(
                self,
                state: AgentState,
                decision: AgentDecision,
        ) -> None:
            """
            Execute one validated planner tool decision and record the resulting
            observation.
            """

            capability = decision.capability

            if capability is None:

                state.failure_count += 1

                state.status = AgentStatus.FAILED

                state.final_answer = (
                    "Planner produced a tool decision "
                    "without a capability."
                ) 

                return
            
            step = AgentStep(
                number=state.step_count + 1,
                decision=decision,
            )

            #-------------------------------------------------------
            # Execute through copntrolled capability boundary
            #-------------------------------------------------------

            observation = self.tools.execute(
                capability,
                decision.arguments,
            )

            step.observation = observation

            state.steps.append(step)

            #-------------------------------------------------------
            # Confirmation gate
            #-------------------------------------------------------

            if (
                observation.requires_confirmation and not
                observation.success and
                observation.status == "blocked"
            ):
                
                state.status = AgentStatus.WAITING_FOR_CONFIRMATION

                state.final_answer =(
                    observation.message 
                    or(
                        "This action requires "
                        "your confirmation."
                    )
                )

                return
            
            #------------------------------------------------------
            # Normal execution failure
            #------------------------------------------------------
            if not observation.success:

                state.failure_count += 1

                # Do not immediately kill the run.

                # The next planner cycle is allowed to see the failed
                #observation and choose a differernt strategy.

                # Hard Failure_count limits still protect us.

                if state.failure_count >= state.max_failures:
                    state.status = AgentStatus.FAILED

                    state.final_answer = (
                        observation.error or
                        observation.message 
                        or (
                            "The agent reached the "
                            "maximum failure limit."
                        )
                    )
                return
            
            #---------------------------------------------------
            # Successful tool execution
            #---------------------------------------------------

            state.status = AgentStatus.RUNNING

    def run(
            self,
            state: AgentState,
    ) -> AgentState:
        """
        Run until Stella reaches a stopping conditiion.

        Returns the updated AgentState.
        """

        if not state.can_continue():
            return state
        
        state.status = AgentStatus.RUNNING

        while state.can_continue():

            #-------------------------------------------------
            # 1. Decide
            #-------------------------------------------------

            try:
                decision = (
                    self.planner.decide(
                        state
                    )
                )

            except AgentPlannerError as exc:

                state.failure_count += 1

                state.status = AgentStatus.FAILED

                state.final_answer = (
                    "The agent planner could not "
                    f"produce a valid next step: {exc}"    
                    )
                
                return state
            
            #--------------------------------------------------
            # 2;. Tool
            #--------------------------------------------------

            if decision.decision_type == AgentDecisionType.TOOL:
                self._run_tool_step(
                    state,
                    decision,
                )

                if state.status != AgentStatus.RUNNING:
                    return state
                
                continue

            #--------------------------------------------------
            # 3. Complete
            #--------------------------------------------------

            if decision.decision_type == AgentDecisionType.COMPLETE:
                state.status = AgentStatus.COMPLETED

                state.final_answer = (
                    decision.message
                    or (
                        "The requested goal has been completed."
                    )
                )

                return state
            
            #---------------------------------------------------
            # 4. Ask User
            #---------------------------------------------------

            if decision.decision_type == AgentDecisionType.ASK_USER:
                state.status = AgentStatus.WAITING_FOR_USER
                
                state.final_answer = (
                    decision.message
                    or (
                        "More information is required before I can continue."
                    )     
                )

                return state
            
            #---------------------------------------------------
            # 5. RESPOND
            #---------------------------------------------------

            if decision.decision_type == AgentDecisionType.RESPOND:

                state.status = AgentStatus.WAITING_FOR_USER

                state.final_answer = decision.message or ""

                return state
            
            #--------------------------------------------------
            # 6. Abort
            #--------------------------------------------------

            if decision.decision_type == AgentDecisionType.ABORT:
                
                state.status = AgentStatus.ABORTED

                state.final_answer = (
                    decision.message
                    or "The agent could not safely continue."
                )

                return state

            #-------------------------------------------------
            # Hard Safety Limits
            #-------------------------------------------------

        if state.failure_count >= state.max_failures:
            state.status = AgentStatus.FAILED

            state.final_answer = (
                "The agent stopped after reaching "
                 "the maximum number of failures."
                )
            
        else:
            state.status = AgentStatus.FAILED
            state.final_answer = (
                "The agent stopped after reaching the " \
                "maximum number of steps."
            )            
        return state
        
        
