from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from state import PhraseGeneratorState
from tools.generation_tools import generate_sentences, generator_node
from tools.handoff_tools import transfer_to_chatbot

SENTENCE_EXPERT_SYSINT = """
    You are a Sentence Expert.
    You must call the generate_sentences tool to generate sentences as soon as the conversation starts.
    After the generation, you should transfer the conversation back to the chatbot.
    \n\n
    If any of the tools are unavailable, you can break the fourth wall and tell the user that 
    they have not implemented them yet and should keep reading to do so.
    """

tools = [transfer_to_chatbot]
generator_tools = [generate_sentences]

def make_sentence_expert(llm: ChatGoogleGenerativeAI):
    agent_tools = tools + generator_tools
    llm_with_tools = llm.bind_tools(agent_tools)
    agent_tools_node = ToolNode(agent_tools)

    def maybe_route_to_tools(state: PhraseGeneratorState) -> str:
        """Route between chat and tool nodes if a tool call is made."""
        if not (msgs := state.get("messages", [])):
            raise ValueError(f"No messages found when parsing state: {state}")

        msg = msgs[-1]

        if state.get("finished", False):
            # When an order is placed, exit the app. The system instruction indicates
            # that the chatbot should say thanks and goodbye at this point, so we can exit
            # cleanly.
            return END

        elif hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
            # Route to `tools` node for any automated tool calls first.
            if msg.tool_calls[-1]["name"] in ToolNode(generator_tools).tools_by_name.keys():
                return "generator"
            else:
                return "tools"

        else:
            return "human"

    def expert(state: PhraseGeneratorState) -> PhraseGeneratorState:
        """The chatbot itself. A wrapper around the model's own chat interface."""

        system_message = SystemMessage(content=SENTENCE_EXPERT_SYSINT)
        new_output = llm_with_tools.invoke([system_message] + state["messages"])

        return state | {"messages": [new_output]}

    def human_node(state: PhraseGeneratorState) -> PhraseGeneratorState:
        """Display the last model message to the user, and receive the user's input."""
        last_msg = state["messages"][-1]
        print("Model:", last_msg.content)

        user_input = input("User: ")

        # If it looks like the user is trying to quit, flag the conversation
        # as over.
        if user_input in {"q", "quit", "exit", "goodbye"}:
            state["finished"] = True

        return state | {"messages": [("user", user_input)]}

    def maybe_exit_human_node(state: PhraseGeneratorState) -> Literal["expert", "__end__"]:
        """Route to the chatbot, unless it looks like the user is exiting."""
        if state.get("finished", False):
            return END
        else:
            return "expert"

    # Start building a new graph.
    graph_builder = StateGraph(PhraseGeneratorState)

    # Add the chatbot and human nodes to the app graph.
    graph_builder.add_node("expert", expert)
    graph_builder.add_node("human", human_node)
    graph_builder.add_node("tools", agent_tools_node)
    graph_builder.add_node("generator", generator_node)

    # Chatbot may go to tools, or human.
    graph_builder.add_conditional_edges("expert", maybe_route_to_tools)
    # Human may go back to chatbot, or exit.
    graph_builder.add_conditional_edges("human", maybe_exit_human_node)

    # Tools always route back to chat afterwards.
    graph_builder.add_edge("tools", "expert")
    graph_builder.add_edge("generator", "expert")

    graph_builder.add_edge(START, "expert")

    return graph_builder.compile()
