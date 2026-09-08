"""
Diagnostic test for Stella Agent Phase 6A

This doesn't run an autonomus agent yet.

It verifies that:

1. Agent models work.
2. Exisiting capabilities are visible to the agent.
3. Capability metadata survives the adapter.
4. Execution still goes through Stella's controlled executor.
5. Results are converted into AgentObservation objects.

"""

from core.agent import(
    AgentGoal,
    AgentState,
    AgentToolAdapter,
)

def print_separator(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():

    #----------------------------------------------------
    # 1. Create an agent goal
    #----------------------------------------------------

    print_separator(
        "AGENT STATE TEST"
    )

    state = AgentState(
        goal=AgentGoal(
            text=(
                "Inspect the current Mac state "
                "and eventually perform a task."
            )
        )
    )

    print(
        "Goal:", state.goal.text,
    )
    print(
        "Status:",
        state.status.value,
    )
    print(
        "Can continue:",
        state.can_continue(), 
    )
    
    #----------------------------------------------------
    # 2. Load capability Adapter
    #----------------------------------------------------

    print_separator(
        "AGENT TOOL INVENTORY"
    )

    tools = AgentToolAdapter()

    available = tools.list_tools()

    print(
        f"Agent sees {len(available)} tools."
    )
    
    for tool in available:

        confirmation = (
            "CONFIRM"
            if tool.requires_confirmation
            else "AUTO"
        )

        print(
            f"{tool.name:<28}"
            f" family={tool.family:<15}"
            f" risk={tool.risk:<12}"
            f" {confirmation}"
        )

        #----------------------------------------------------
        # 3. Inspect one tool
        #----------------------------------------------------

        print_separator(
            "SINGLE TOOL METADATA"
        )

        tool = tools.get_tool(
            "get_system_volume"
        )

        print(tool)

        #----------------------------------------------------
        # 4. Execute a SAFE read-only capability
        #----------------------------------------------------

        print_separator(
            "AGENT TOOL EXECUTION"
        )

        observation = tools.execute(
            "get_system_volume",
        )

        print(
            "Capability:",
            observation.capability,
        )

        print(
            "Success:",
            observation.success,
        )

        print(
            "Status:",
            observation.status,
        )
        print(
            "Data:",
            observation.data,
        )
        print(
            "Message:",
            observation.message,
        )
        print(
            "Error:",
            observation.error,
        )
        print(
            "Requires confirmation:",
            observation.requires_confirmation,
        )

        print_separator(
            "DONE"
        )

if __name__ == "__main__":
    main()