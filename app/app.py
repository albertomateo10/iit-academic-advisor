import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import os
import sys
# Import your compiled LangGraph agent from your agents folder
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from agents.graph import advisor_agent

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IIT Academic Advisor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 IIT Proactive Academic Advisor")
st.markdown("Welcome! I can help you find courses, check prerequisites, and plan your ITM degree.")

# --- SESSION STATE INITIALIZATION ---
# This ensures the LangGraph memory survives when Streamlit reruns the page
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "messages": [],
        "completed_courses": [],
        "current_gpa": 4.0,
        "awaiting_prereq_confirmation": False
    }

# --- RENDER CHAT HISTORY ---
# We loop through the messages in the LangGraph state and display them.
for msg in st.session_state.agent_state["messages"]:
    # We only want to show the Human and AI messages to the user.
    # We HIDE the System messages and Tool outputs so the UI stays clean!
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        # Some AI messages are just blank tool calls. We only render ones with text.
        with st.chat_message("assistant"):
            st.write(msg.content)

# --- HANDLE NEW USER INPUT ---
# st.chat_input creates the text box at the bottom of the screen
if prompt := st.chat_input("Ask me about IIT courses..."):
    
    # 1. Display the user's message immediately in the UI
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Add the user's message to our LangGraph state
    st.session_state.agent_state["messages"].append(HumanMessage(content=prompt))

    # 3. Trigger the LangGraph Agent
    with st.spinner("Searching the IIT catalog..."):
        # We pass the current state into the graph and get the updated state back
        new_state = advisor_agent.invoke(st.session_state.agent_state)
        
        # Overwrite our Streamlit state with the new LangGraph state (which includes the AI's reply)
        st.session_state.agent_state = new_state

        # 4. Display the AI's response in the UI
        # The AI's final answer is always the last message in the list
        final_message = new_state["messages"][-1]
        
        if isinstance(final_message, AIMessage) and final_message.content:
            with st.chat_message("assistant"):
                st.write(final_message.content)