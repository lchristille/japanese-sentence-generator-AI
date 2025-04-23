from typing_extensions import TypedDict, Annotated

from langgraph.graph import add_messages


class PhraseGeneratorState(TypedDict):
    """State representing the conjugation of a verb."""

    # The chat conversation. This preserves the conversation history
    # between nodes. The add_messages annotation defines how this
    # state key should be updated (in this case, it appends messages
    # to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

    # The verbs to use to generate the sentences for
    verbs: list[str]

    # The tense to generate the sentences for
    tenses: list[str]

    # Flag indicating whether the sentence generation is complete
    finished: bool

    generated_phrases: list[str]