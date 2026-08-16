"""
Diagnostic script 2 - runs the FULL agent exactly as app.py does, but prints every
intermediate message (tool calls requested by the LLM, and the raw result each tool
returned), so we can see exactly where this specific query breaks down.

Run with: python Test2.py
Works whether placed next to 'agents' or one level up, same as the previous script.
"""
import sys, os

_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.abspath(os.path.join(_here, '..'))
for _path in (_here, _parent):
    if _path not in sys.path:
        sys.path.append(_path)

if not os.path.isdir(os.path.join(_here, 'agents')) and not os.path.isdir(os.path.join(_parent, 'agents')):
    print(f"WARNING: could not find an 'agents' folder in {_here} or {_parent}.")
    print("Move this script so that 'agents' is either next to it or one level up, then rerun.")

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from agents.graph import advisor_agent

state = {
    "messages": [HumanMessage(content=(
        "I am pursuing the Master of Data Science (with Applied Mathematics). "
        "I have completed MATH 563, SCI 522, CSP 571, and MATH 474. "
        "What core course should I take next?"
    ))],
    "completed_courses": ["MATH 563", "SCI 522", "CSP 571", "MATH 474"],
    "current_gpa": 4.0,
    "awaiting_prereq_confirmation": False,
}

print("Invoking the agent (streaming, limit raised to 20 just for this diagnostic)...\n")

step_num = 0
try:
    for step in advisor_agent.stream(state, config={"recursion_limit": 20}, stream_mode="values"):
        step_num += 1
        messages = step["messages"]
        last_msg = messages[-1]
        print(f"\n--- Step {step_num}: {type(last_msg).__name__} ---")

        tool_calls = getattr(last_msg, "tool_calls", None)
        if tool_calls:
            print("TOOL CALLS REQUESTED BY THE LLM:")
            for tc in tool_calls:
                print(f"  -> {tc.get('name')}(args={tc.get('args')})")

        name = getattr(last_msg, "name", None)
        if name:
            print(f"(this is the RESULT of tool: {name})")

        content = getattr(last_msg, "content", None)
        if content:
            print("CONTENT:")
            print(content)

except GraphRecursionError:
    print("\n" + "=" * 70)
    print(f"*** Hit the recursion limit after {step_num} steps, still no final answer. ***")
    print("Look at the pattern above: is the same tool being called over and over with")
    print("similar arguments, or is it alternating between different tools without ever")
    print("producing a plain-text final message?")

print("\n" + "=" * 70)
print("Done.")