from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import Literal

from state import PhraseGeneratorState
from tools.collection_tools import get_verb_tenses, is_verb, add_verb, clear_verbs, add_tense, clear_tenses, \
    get_notebook, confirm_generation, collector_node
from tools.handoff_tools import transfer_to_sentence_expert

CHATBOT_SYSINT = (
    "system",  # 'system' indicates the message is a system instruction.
    "You are a Chatbot of an interactive phrase generator system. You can talk in English."
    "A human will talk to you about the different tenses that a verb can have and you will answer any questions about verb tenses "
    "(and only about verb tenses - no off-topic discussion, but you can chat about the verb meaning). "
    "The user will choose verbs to generate phrases for, and the tenses he wants. "
    "After that you will generate phrases for those verbs in that tenses."
    "You cannot generate phrases if you do not have at least one verb and one tense."
    "\n\n"
    "Add the chosen verb to the user notebook with add_verb, and reset the verbs with clear_verb."
    "Add tenses to the user notebook with add_tense, and reset the tenses with clear_tenses. "
    "To see the contents of the notebook so far, call get_notebook (this is shown to you, not the user) "
    "Always confirm_generation with the user (double-check) before calling generate_phrases. Calling confirm_generation will "
    "display the selected items to the user and returns their response to seeing the list. Their response may contain modifications. "
    "Always verify and respond with tenses from the GRAMMAR before adding them to the notebook. "
    "Always verify for each chosen word if it's a verb with is_verb."
    "You can only generate phrases starting from verbs. "
    "Once the user has finished composing its notebook, Call confirm_generation to ensure it is correct then make "
    "any necessary updates."
    "You can ask the sentence expert for help with generating phrases."
    "\n\n"
    "Do not mention the tools name in responses."
    "If any of the tools are unavailable, you can break the fourth wall and tell the user that "
    "they have not implemented them yet and should keep reading to do so.",
)

WELCOME_MSG = "Welcome to the PhraseGenerator dojo. Type `q` to quit. What phrases can I generate today?"

tools = [get_verb_tenses, is_verb, transfer_to_sentence_expert]
collection_tools = [add_verb, clear_verbs, add_tense, clear_tenses, get_notebook, confirm_generation]


def make_chatbot(llm: ChatGoogleGenerativeAI):
    agent_tools = tools + collection_tools
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
            if msg.tool_calls[-1]["name"] in ToolNode(collection_tools).tools_by_name.keys():
                return "collector"
            else:
                return "tools"

        else:
            return "human"

    def chatbot(state: PhraseGeneratorState) -> PhraseGeneratorState:
        """The chatbot itself. A wrapper around the model's own chat interface."""

        if state["messages"]:
            # If there are messages, continue the conversation with the Gemini model.
            new_output = llm_with_tools.invoke([CHATBOT_SYSINT] + state["messages"])
        else:
            # If there are no messages, start with the welcome message.
            new_output = AIMessage(content=WELCOME_MSG)

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

    def maybe_exit_human_node(state: PhraseGeneratorState) -> Literal["chatbot", "__end__"]:
        """Route to the chatbot, unless it looks like the user is exiting."""
        if state.get("finished", False):
            return END
        else:
            return "chatbot"

    # Start building a new graph.
    graph_builder = StateGraph(PhraseGeneratorState)

    # Add the chatbot and human nodes to the app graph.
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("human", human_node)
    graph_builder.add_node("tools", agent_tools_node)
    graph_builder.add_node("collector", collector_node)

    # Chatbot may go to tools, or human.
    graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
    # Human may go back to chatbot, or exit.
    graph_builder.add_conditional_edges("human", maybe_exit_human_node)

    # Tools always route back to chat afterwards.
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge("collector", "chatbot")

    graph_builder.add_edge(START, "chatbot")

    return graph_builder.compile()
