from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import Annotated

@tool
def transfer_to_sentence_expert(state: Annotated[dict, InjectedState],
                                tool_call_id: Annotated[str, InjectedToolCallId]):
    """Transfer the conversation to the sentence expert."""

    state['messages'].append(
        ToolMessage(
            content="Transferring conversation to sentence expert.",
            name="transfer_to_sentence_expert",
            tool_call_id=tool_call_id
        )
    )

    return Command(
        goto="sentence_expert",
        graph=Command.PARENT,
        update=state
    )

@tool
def transfer_to_chatbot(state: Annotated[dict, InjectedState],
                                tool_call_id: Annotated[str, InjectedToolCallId]):
    """Transfer the conversation to the sentence expert."""

    state['messages'].append(
        ToolMessage(
            content="Transferring conversation to chatbot.",
            name="transfer_to_chatbot",
            tool_call_id=tool_call_id
        )
    )
    return Command(
        goto="chatbot",
        graph=Command.PARENT,
        update=state
    )