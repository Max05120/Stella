"""
core/agent/planner.py

Goal-oriented decision planner for Stella's agent runtime.

The planner does NOT execute tools.

It's only responsibility is:

    AgentState
        +
    available capabilities
        ↓
    one structures AgentDecision

The planner deliberately decides one meaningful next step at a time.
"""

from __future__ import annotations

import json
from typing import Any
import ollama

from core.agent.models import (
    AgentDecision,
    AgentDecisionType,
    AgentState,
)

from core.agent.tool_adapter import (
    AgentTool,
    AgentToolAdapter,
)

from core.config import (
    LLM_MODEL,
    OLLAMA_KEEP_ALIVE,
)

AGENT_PLANNER_SYSTEM_PROMPT = """
You are the planning component of Stella, a macOS agent.

Your job is NOT to directly answer the user and NOT to execute anything.

Your job is to decide the single best NEXT STEP toward accomplishing the user's goal.

You have access only to the capabilities explicitly provided to you.

IMPORTANT RULES:

1. Decide exactly ONE next step.

2. Never invent capabilities names.

3. If a tool is needed, use the exact capability name for the 
AVAILABLE CAPABILITIES list.

4. For a tool decision, arguments must contain ONLY parameters shown in
 that capability's PARAMETERS section.

5. Do not copy metadata such as family, risk, confirmation, permissions,
or description into arguments.

6. Every required parameter must be supplied.

7. Never invent usernames or absolute home-directory paths

8. When referring to the user's home directory, prefer "~" when supported
by the capability.

9. Do not attempt to open a file or folder that must first be created.
Choose the creation capability first.

10. Never invent observations or claim an action succeeded before recieveing
an observation.

11. Prefer inspecting the environment before making assumptions.

12. If the goal is already satisfied based on observations, choose 
decision_type "complete".

13. If progress requires information only the user can provide, choose
decision_type "ask_user".

14. If the goal cannot safely or reasonably be completed, choose
decision_type "abort".

15. "respond" is for communicating something useful without claiming the overall
goal is complete.

16. Do not produce executable code, shell commands, AppleScript, Python, or 
arbitrary instructions without concern. You may only select one of Stella's 
registered capabilities, if there's no possiblities withing registered capabilities
you can suggest the alternative to the user.

17. Do not bypass confirmation requirements. The execution layer handles
confirmation.

18. Keep reasoning_summary short. It should explain the decision without 
exposing long internal reasoning.

19. For every required tool parameter, extract or derive the value from
the user's goal or previous observations and include it in arguments.

20. A required parameter must never be omitted merely because it's meaning
seems obvious.

Successful observation are facts.

When a tool succeeds, treat the part of the goal accomplished by that
tool as completed.

Do not repeat the same successful capability with the same arguments unless
a later observation proves that repeating it is necessary.

For multi-part goals, advance to the next unsatisfied requirement after each successful observation.

Before selecting a tool, inspect previous successful steps and ask:
"What part of the user's goal is still unfinished?"

If all parts of the goal are satisfied by previous observations,
choose COMPLETE instead of calling another tool.

When completing an information-gathering goal, include the relevant observed
information in the COMPLETE message.

When an information-gathering step succeeds, use its returned DATA
when completing the task.

Do not perform another state-changing action merely because an
observation returned an unexpected answer.

If the user asks you to report something and the relevant observation
already contains the answer, choose COMPLETE and report that answer.

FAILURE RECOVERY RULES:

When a previous tool failed, inspect its RECOVERY_TYPE,
RETRY_SAME_ACTION, and RECOVERY_GUIDANCE before deciding what to do.

A failed tool call is an observation about the real environment.
Do not ignore it.

If RETRY_SAME_ACTION is False, do not issue the same capability with
the same arguments again.

For NOT_FOUND failures, prefer discovery or inspection tools that can
resolve the correct resource before trying another action.

For ALREADY_EXISTS failures, treat the existing resource as potentially
satisfying only the creation requirement that failed.

Then re-read the ORIGINAL GOAL and continue with every remaining
unsatisfied requirement.

Example:
If the goal is "Create folder X and then open it", and create_folder
reports ALREADY_EXISTS, the creation requirement may already be
satisfied, but the open requirement is NOT satisfied.

The correct next action is to open the existing folder.
Do NOT choose COMPLETE until the remaining requirement is satisfied.

For PERMISSION_DENIED failures, do not repeatedly retry the blocked
operation. Use another valid strategy or explain the permission issue.

For UNKNOWN failures, prefer a different safe strategy, ASK_USER,
or ABORT rather than blindly repeating the failed call.

A recovery action must still advance the original user goal.
Do not perform unrelated actions merely because a tool failed.
Example:

Goal:
Open Spotify and tell me what application is frontmost.

History:
open_application Spotify -> SUCCESS
get_frontmost_application -> SUCCESS
DATA: {"name": "Code"}

Correct next decision:

{
    "decision_type": "complete",
    "capability": null,
    "arguments": {},
    "message": "Code is currently the frontmost application.",
    "reasoning_summary": null
}

Do not reopen Spotify simply because the observed frontmost
application was not Spotify.

Example:

User goal:
Open Spotify.

Available capability:
open_application
Parameters:
-   application: str (required)

Correct decision:
{
    "decision_type": "tool",
    "capability": "open_application",
    "arguments": {
        "application": "Spotify"
    },
    "message": null,
    "reasoning_summary": null
}

Example:

User goal:
Create a folder called Agent Test in my Downloads folder and open it.

The folder does not exist yet, so the first decision must be:

{
    "decision_type": "tool",
    "capability": "create_folder",
    "arguments": {
        "parent_path": "~/Downloads",
        "folder_name": "Agent Test"
    },
    "message": null,
    "reasoning_summary": null
}

Do NOT choose open_path until an observation confirms that the folder was
successfully created.

You MUST return valid JSON only.

The JSON object must have exactly these fields:
{
    "decision_type": "tool | respond | ask_user | complete | abort",
    "capability": "exactly capability name or null",
    "arguments": {},
    "message": "user-facing message or null",
    "reasoning_summary": "short explanation or null"
}

For a tool decision:
{
    "decision_type": "tool",
    "capability": "get_system_volume",
    "arguments": {},
    "message": null,
    "reasoning_summary": "Inspect the current volume before continuing."
}

For completion:
{
    "decision_type": "complete",
    "capability": null,
    "arguments": {},
    "message": "The requested task is complete."
    "reasoning_summary": "The observations show the goal has been satisfied."
}

For decision_type "tool", normally set "message" to null.
The execution layer will report what actually happened.`

""".strip()

class AgentPlannerError(Exception):
    """
    Raisedd when the planner cannot produce a safe,
    valid AgentDecision.
    """

class AgentPlanner:
    """
    Produces one structured next-step decision for an AgentState.
    """
    def __init__(
            self,
            tool_adapter: AgentToolAdapter | None = None,
    ):
        self.tools =(
            tool_adapter
            or AgentToolAdapter()
        )

    def decide(
            self,
            state: AgentState,
    ) -> AgentDecision:
        """
        Decide the single best next action for the current agent state.

        If the model produces a structurally invalid decision,
        allow one repair attempt using the validation error.
        """

        available_tools = (
            self.tools.list_tools()
        )

        prompt = self._build_prompt(
            state,
            available_tools,
        )

        #---------------------------------------------------
        # First planning attempt
        #---------------------------------------------------

        raw_content= self._call_model(
            prompt
        )

        try:
            decision = self._parse_decision(
                raw_content,
                available_tools,
            )
        
            self._validate_progress(
                state,
                decision,
            )

            self._validate_completion(
                state,
                decision,
            )

            return decision
        
        except AgentPlannerError as first_error:

            print(
                "[AGENT PLANNER] "
                f"decision rejected: {first_error}"
            )

            print(
                "[AGENT PLANNER] "
                f"attempting one repair"
            )
        #---------------------------------------------------
        # One controlled repair attempt
        #---------------------------------------------------

            repair_prompt =(
                self._build_repair_prompt(
                    state=state,
                    tools=available_tools,
                    previous_output=raw_content,
                    validation_error=str(
                        first_error
                    ),
                )
            )

            repaired_content = (
                self._call_model(
                    repair_prompt
                )
            )

            try:
                decision = self._parse_decision(
                    repaired_content,
                    available_tools,
                )
                self._validate_progress(
                    state,
                    decision,
                )
                self._validate_completion(
                    state,
                    decision,
                )
                print(
                    "[AGENT PLANNER] "
                    "repair successful"
                )

                return decision
                # return self._parse_decision(
                #     repaired_content,
                #     available_tools,
                # )
            
            except AgentPlannerError as second_error:
                raise AgentPlannerError(
                    (
                        "Planner failed after one repair attempt."
                        f"Initial error: {first_error}."
                        f"Repair error: {second_error}"
                    )
                ) from second_error
            
    def _validate_progress(
            self,
            state: AgentState,
            decision: AgentDecision,
    ) -> None:
        """
        Prevent the planner from immediately repeating an action that 
        already succeeded with the same arguments.
        """
        if decision.decision_type != AgentDecisionType.TOOL:
            return
        latest_step = state.latest_step

        capability = decision.capability

        if capability is None:
            return
        #----------------------------------------------------------
        # Prevent pointless repetition of non-retryable failures.
        #----------------------------------------------------------
        for step in reversed(
            state.steps
        ):
            observation = step.observation

            if observation is None:
                continue

            if observation.success:
                continue

            previous = step.decision

            if previous.decision_type != AgentDecisionType.TOOL:
                continue

            same_capability = (
                previous.capability == capability
                )
            
            same_arguments = (
                previous.arguments == decision.arguments
            )

            if not (
                same_capability and same_arguments
            ):
                continue
            if (
                observation.retry_same_action is True
            ):
                # One retry is allowed for failures classified
                # as potentially transient.
                identical_failures = 0

                for previous_step in state.steps:

                    previous_observation = previous_step.observation

                    if (
                        previous_observation is None or
                        previous_observation.success
                    ):
                        continue
                    previous_decision = previous_step.decision

                    if (
                        previous_decision.capability ==
                        capability and previous_decision.arguments == 
                        decision.arguments
                    ):
                        identical_failures += 1

                if identical_failures < 2:
                    break

            guidance =(
                observation.recovery_guidance 
                or (
                    "Choose a differernt strategy"
                    " instead of repeating the same "
                    "failed action."
                )
            )

            raise AgentPlannerError(
                (
                    "This exact tool call already failed " \
                    "and has been calssified as " \
                    "non-retryable. " \
                    f"Recovery guidance: {guidance}"
                )
            )

        # ---------------------------------------------------------
        # Observation tools
        #
        # These inspect state rather than changing it.
        # ---------------------------------------------------------

        observational = (
            capability.startswith("get_")
            or capability.startswith("list_")
        )

        # ---------------------------------------------------------
        # Find previous successful identical calls
        # ---------------------------------------------------------

        previous_match_index = None

        for index, step in enumerate(
            state.steps
        ):

            observation = step.observation

            if observation is None:
                continue

            if not observation.success:
                continue

            previous = step.decision

            if (
                previous.decision_type
                != AgentDecisionType.TOOL
            ):
                continue

            same_capability = (
                previous.capability
                == capability
            )

            same_arguments = (
                previous.arguments
                == decision.arguments
            )

            if (
                same_capability
                and same_arguments
            ):
                previous_match_index = index

        # Never executed successfully before.
        if previous_match_index is None:
            return

        # ---------------------------------------------------------
        # Side-effecting action:
        #
        # create_folder
        # open_application
        # open_path
        # move_file
        # etc.
        #
        # Once successful, don't repeat the exact same operation
        # during this goal.
        # ---------------------------------------------------------

        if not observational:

            raise AgentPlannerError(
                (
                    "This exact action already succeeded "
                    "earlier in this agent run. "
                    "Do not repeat it. Treat that successful "
                    "result as completed and continue with "
                    "the remaining unsatisfied part of the goal, "
                    "or choose COMPLETE if nothing remains."
                )
            )

        # ---------------------------------------------------------
        # Observation action:
        #
        # It MAY be repeated, but only if something changed
        # after the previous observation.
        # ---------------------------------------------------------

        later_steps = (
            state.steps[
                previous_match_index + 1:
            ]
        )

        state_changed = False

        for later_step in later_steps:

            later_observation = (
                later_step.observation
            )

            if (
                later_observation is None
                or not later_observation.success
            ):
                continue

            later_capability = (
                later_step.decision.capability
                or ""
            )

            later_is_observational = (
                later_capability.startswith("get_")
                or later_capability.startswith("list_")
            )

            if not later_is_observational:
                state_changed = True
                break

        if not state_changed:

            raise AgentPlannerError(
                (
                    "This observation was already performed "
                    "successfully and no state-changing action "
                    "has happened since then. "
                    "Use the existing observation instead of "
                    "querying it again, or choose COMPLETE."
                )
            )
    
    def _validate_completion(
    self,
    state: AgentState,
    decision: AgentDecision,
    ) -> None:
        """
        Reject obviously unsupported COMPLETE decisions.

        This is deliberately conservative. It does not try to
        fully understand the user's goal deterministically.
        Instead, it catches completion immediately after a
        failed action when no subsequent successful action has
        demonstrated progress.
        """

        if (
            decision.decision_type
            != AgentDecisionType.COMPLETE
        ):
            return

        if not state.steps:
            return

        latest_step = state.steps[-1]
        observation = latest_step.observation

        if observation is None:
            raise AgentPlannerError(
                (
                    "COMPLETE is not justified because the "
                    "latest tool step has no observation."
                )
            )

        if observation.success:
            return

        # -----------------------------------------------------
        # A failed action may still reveal that one subgoal is
        # already satisfied, such as create_folder reporting
        # ALREADY_EXISTS.
        #
        # That does NOT prove that the entire multi-part goal
        # has been completed.
        # -----------------------------------------------------

        if (
            observation.recovery_type
            == "already_exists"
        ):
            raise AgentPlannerError(
                (
                    "COMPLETE is not yet justified. "
                    "The latest action failed because the "
                    "requested resource already exists. "
                    "That may satisfy only the creation part "
                    "of the goal. Re-read the ORIGINAL GOAL "
                    "and identify whether any later requirement "
                    "is still unfinished. If another requested "
                    "action remains, perform it now. Choose "
                    "COMPLETE only when every part of the "
                    "original goal is supported by the "
                    "observed history."
                )
            )

        raise AgentPlannerError(
            (
                "COMPLETE is not justified immediately after "
                "a failed tool execution. Re-read the original "
                "goal and the recovery guidance. Choose another "
                "valid action, ASK_USER, or ABORT unless the "
                "history actually demonstrates that every "
                "requested requirement is satisfied."
            )
        )

    def _call_model(
            self,
            prompt: str,
    ) -> str:
        """
        Run one planner inference and return the raw JSON text.
        """

        response = ollama.chat(
            model=LLM_MODEL,
            messages= [
                {
                    "role": "system",
                    "content": (
                        AGENT_PLANNER_SYSTEM_PROMPT
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            format="json",
            keep_alive=OLLAMA_KEEP_ALIVE,
            options={
                "temperature": 0.1,
                "num_predict": 300,
            },
        )

        return response[
            "message"
        ][
            "content"
        ]


    def _build_prompt(
            self,
            state: AgentState,
            tools: list[AgentTool],
    ) -> str:
        """
        Build a compact planning prompt from current state.
        """

        tool_text = self._format_tools(
            tools
        )

        history_text = (
            self._format_history(
                state
            )
        )
        return f"""
USER GOAL
---------
{state.goal.text}

CURRENT AGENT STATUS
--------------------
{state.status.value}

Steps used:
{state.step_count} / {state.max_steps}

Failures:
{state.failure_count} / {state.max_failures}

PREVIOUS STEPS AND OBSERVATIONS
-------------------------------
{history_text}

AVAILABLE CAPABILITIES
----------------------
{tool_text}

Decide the single best NEXT STEP toward the goal.

Return valid JSON only.
""".strip()
    
    def _build_repair_prompt(
            self,
            state: AgentState,
            tools: list[AgentTool],
            previous_output: str,
            validation_error: str,
    ) -> str:
        """
        Ask the planner to correct one invalid decision.

        The planner is given it's previous output together with the exact
        validation failure.

        It must resturn a complete replacement decision.
        """

        tool_text = self._format_tools(
            tools
        )

        history_text = (
            self._format_history(
                state
            )
        )

        return f"""
Your previous decision was rejected by Stella's validator.

USER GOAL
--------
{state.goal.text}

PREVIOUS STEPS AND OBSERVATIONS
-------------------------------
{history_text}

YOUR REJECTED OUTPUT
--------------------
{previous_output}

VALIDATION ERROR
----------------
{validation_error}

AVAILABLE CAPABILITIES AND EXACT PARAMETERS

AVAILABLE CAPABILITIES AND EXACT PARAMETERS
-------------------------------------------
{tool_text}

Correct the decision.

IMPORTANT:

- Return one full replacement JSON decision object.
- "full replacement JSON decision object" does NOT mean decision_type "complete".
- The replacement decision must correct the exact VALIDATION ERROR above.
- Do NOT repeat a decision that the validator just rejected.
- Re-read the ORIGINAL USER GOAL and identify what requirement is still unfinished.
- Inspect PREVIOUS STEPS AND OBSERVATIONS before choosing the replacement decision.
- If a failed observation contains RECOVERY_TYPE, RETRY_SAME_ACTION, or
  RECOVERY_GUIDANCE, obey that recovery information.
- If ALREADY_EXISTS shows that a creation requirement is already satisfied,
  do not create the resource again. Continue with the NEXT unfinished
  requirement using the existing resource.
- If the validation error says COMPLETE is not justified, decision_type
  MUST NOT be "complete" in the repaired decision.
- Use exactly one capability when decision_type is "tool".
- Use ONLY parameter names listed for that capability.
- Supply every required parameter.
- Arguments must contain values needed to execute the action.
- Do not put family, risk, confirmation, description, or permissions
  inside arguments.
- Never invent a username.
- Use "~" for the user's home directory when appropriate.
- Return valid JSON only.
""".strip()



    def _format_tools(
            self,
            tools: list[AgentTool],
    ) -> str:
        """
        Format capabilities together with their real argument schemas.
        """

        blocks: list[str] = []
        for tool in tools:
            
            confirmation = (
                "yes"
                if tool.requires_confirmation
                else "no"
            )

            lines = [
                f"CAPABILITY: {tool.name}",
                f"DESCRIPTION: {tool.description}",
                f"FAMILY: {tool.family}",
                f"RISK: {tool.risk}",
                (
                    "REQUIRES_CONFIRMATION: "
                    f"{confirmation}"
                ),
                "PARAMETERS:",
            ]

            if not tool.parameters:
                lines.append(
                    " none"
                )
            else:
                for parameter in (
                    tool.parameters
                ):
                    required = (
                        "required"
                        if parameter.required
                        else (
                            "optional, "
                            f"default={parameter.default!r}"
                        )
                    )

                    lines.append(
                        (
                            f" - {parameter.name}"
                            f"{parameter.type_name} "
                            f"({required})"
                        )
                    )
            blocks.append(
                "\n".join(lines)
            )
        return "\n\n".join(blocks)
        
    def _format_history(
            self,
            state: AgentState,
    ) -> str:
        """
        Convert previous decisions and observations
        into concise planning context.
        """

        if not state.steps:
            return "No actions have been taken yet."
        
        lines = []

        for step in state.steps:
            decision = step.decision

            lines.append(
                    f"STEP: {step.number}: "
            )
            lines.append(
                (
                    "DECISION: "
                    f"{decision.decision_type.value}"
                )
            )
            lines.append(
                "CAPABILITY: "
                f"{decision.capability}"
            )
            lines.append(f"ARGUMENTS: {decision.arguments}")
            
            observation = (
                step.observation
            )

            if observation is None:
                lines.append(
                    "OBSERVATION = None"
                )
            else:
                if observation.success:
                    lines.append(
                        "RESULT: SUCCESS"
                    )

                    lines.append(
                        (
                            "IMPORTANT: This action " \
                            "already succeeded. Do not " \
                            "repeat it unnecessarily."
                        )
                    )
                else:
                    lines.append(
                        "RESULT: FAILURE"
                    )
                lines.append(
                    (
                        "STATUS: "
                        f"{observation.status}"
                    )
                )

                lines.append(
                    (
                        "DATA: "
                        f"{observation.data}"
                    )
                )
                lines.append(
                    (
                        "MESSAGE: "
                        f"{observation.message}"
                    )
                )
                lines.append(
                    (
                        "ERROR: "
                        f"{observation.error}"
                    )
                )
                if not observation.success:

                    lines.append(
                        (
                            "RECOVERY_TYPE: "
                            f"{observation.recovery_type}"
                        )
                    )

                    lines.append(
                        (
                            "RETRY_SAME_ACTION: "
                            f"{observation.retry_same_action}"
                        )
                    )

                    lines.append(
                        (
                            "RECOVERY_GUIDANCE: "
                            f"{observation.recovery_guidance}"
                        )
                    )
                lines.append("")
        return "\n".join(lines)
    
    def _parse_decision(
            self,
            raw_content: str,
            tools: list[AgentTool],
    ) -> AgentDecision:
        """
        Parse and validate raw LLM JSON.

        Never trust planner output simply because it came from the model.
        """

        try: 
            payload = json.loads(
                raw_content
            )

        except json.JSONDecodeError as exc:
            raise AgentPlannerError(
                "Planner returned invalid JSON."
            )
        
        if not isinstance(
            payload,
            dict,
        ):
            raise AgentPlannerError(
                "Planner response must be a JSON object."
            )
        
        decision_type_raw = (
            payload.get(
                "decision_type"
            )
        )

        try:
            decision_type =(
                AgentDecisionType(
                    decision_type_raw
                )
            )
        except (
            ValueError,
            TypeError,
        ) as exc:
            raise AgentPlannerError(
                (
                    "Planner returned an invalid "
                    f"decision_type: "
                    f"{decision_type_raw!r}"
                )
            )from exc
        
        capability = payload.get(
            "capability"
        )

        arguments = payload.get(
            "arguments",
            {},
        )

        message = payload.get(
            "message"
        )

        reasoning_summary = (
            payload.get(
                "reasoning_summary"
            )
        )

        #------------------------------------------------
        # Validate argument structure
        #------------------------------------------------

        if arguments is None:
            arguments = {}

        if not isinstance(
            arguments,
            dict,
        ):
            raise AgentPlannerError(
                "Planner arguments must be a JSON object."
            )

        #------------------------------------------------
        # Validate tool decision
        #------------------------------------------------

        if (
            decision_type 
            == AgentDecisionType.TOOL 
        ):
            if not isinstance(
                capability,
                str,
            ) or not capability:
                raise AgentPlannerError(
                    (
                        "Tool decision did not "
                        "contain a capability."
                    )
                )
            available_names = {
                tool.name
                for tool in tools
            }

            if (
                capability
                not in available_names
            ):
                raise AgentPlannerError(
                    (
                        "Planner attempted to use "
                        "an unavailable capability: "
                        f" {capability}"
                    )
                )
            is_valid, validation_error = (
                self.tools.validate_arguments(
                    capability,
                    arguments,
                )
            )

            if not is_valid:
                raise AgentPlannerError(
                    (
                        "Planner produced invalid "
                        f"arguments for "
                        f"{capability}: "
                        f"{validation_error}"
                    )
                )
        #--------------------------------------------------
        # Non-tool decision must not carry a capability
        #--------------------------------------------------

        else:
            capability = None
        
        return AgentDecision(
            decision_type=decision_type,
            capability=capability,
            arguments=arguments,
            message=(
                str(message)
                if message is not None
                else None
            ),
            reasoning_summary=(
                str(reasoning_summary)
                if reasoning_summary is not None
                else None
            ),
        )
    
    @staticmethod
    def _safe_json(
        value: Any,
    ) -> str:
        """
        Serialize observations without allowing unexpected
        Python objects to crash prompt construction.
        """

        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )
        except Exception:
            return str(value)
        