import dango
from dango.word import PartOfSpeech
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from state import PhraseGeneratorState

@tool
def get_verb_tenses() -> str:
    """Get a list of all the verb tenses."""

    return """
    GRAMMAR:
    Non-Past Tense:
    Non-Past Affirmative Plain
    Non-Past Affirmative Polite
    Non-Past Negative Plain
    Non-Past Negative Polite

    Past Tense:
    Past Affirmative Plain
    Past Affirmative Polite
    Past Negative Plain
    Past Negative Polite

    Te Form
    Desiderative
    Volitional
    Exhortative
    Imperative
    Request
    Potential
    Passive
    Causative
    Conditional
    """

@tool
def is_verb(word: str):
    """Call to verify that a japanese word is a verb."""
    tokens = dango.tokenize(word)
    if len(tokens) != 1:
        raise ValueError("Input must be a single word.")
    only_word = tokens[0]
    return only_word.part_of_speech == PartOfSpeech.VERB

@tool
def add_verb(verb: str) -> str:
    """Adds the specified verb to the notebook.

    Returns:
        The updated verb list.
    """

@tool
def clear_verbs():
    """Removes the selected verb from the notebook."""

@tool
def add_tense(tense: str) -> str:
    """Adds the specified tense to the notebook.

    Returns:
    The updated verb list.
    """

@tool
def clear_tenses():
    """Removes all tenses from the notebook."""

@tool
def get_notebook():
    """Returns the current elements in the notebook."""

@tool
def confirm_generation():
    """Confirms the generation of sentences."""

def collector_node(state: PhraseGeneratorState) -> PhraseGeneratorState:
    """The collector node. This is where the phrase generator state is manipulated."""
    tool_msg = state.get("messages", [])[-1]
    verbs = state.get("verbs", [])
    tenses = state.get("tenses", [])
    outbound_msgs = []
    generation_complete = False

    for tool_call in tool_msg.tool_calls:
        if tool_call["name"] == "add_verb":
            verbs.append(tool_call["args"]["verb"])
            response = "\n".join(verbs)
        elif tool_call["name"] == "clear_verb":
            verbs.clear()
        elif tool_call["name"] == "add_tense":
            tenses.append(tool_call["args"]["tense"])
            response = "\n".join(tenses)
        elif tool_call["name"] == "clear_tenses":
            tenses.clear()
        elif tool_call["name"] == "get_notebook":
            response = "Verbs:\n" + "\n".join(verbs) + "\n" + "Tenses:\n" + "\n".join(tenses)
        elif tool_call["name"] == "confirm_generation":
            print("Your notebook:")
            print("Verbs:")
            if not verbs:
                print("   (no verbs)")
            for verb in verbs:
                print(f"   {verb}")
            print("Tenses:")
            if not tenses:
                print("   (no tenses)")
            for tense in tenses:
                print(f"   {tense}")
            response = input("Is this correct? (y/n): ")
        else:
            raise NotImplementedError(f'Unknown tool call: {tool_call["name"]}')

        outbound_msgs.append(
            ToolMessage(
                content=response,
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            )
        )

    state = {
        "messages": outbound_msgs,
        "verbs": verbs,
        "finished": False,
        "tenses": tenses,
        "generated_phrases": []
    }

    return state