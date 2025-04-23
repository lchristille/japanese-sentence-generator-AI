import dango
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from state import PhraseGeneratorState

class PhraseResponse(BaseModel):
    """Response model for Japanese phrase generation"""
    verb: str = Field(description="The Japanese verb used in phrase generation")
    tense: str = Field(description="The grammatical tense used in phrase generation")
    sentence: str = Field(description="The generated sentence")

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm = model.with_structured_output(PhraseResponse)

GENERATOR_SYSINT = (
    "system",  # 'system' indicates the message is a system instruction.
    "You are a Japanese Sentence Generator."
    "Given a verb and a tense, you generate a simple sentence that must contain the given verb in the given tense."
    "You cannot do anything else."
    "The sentence should be simple, with words matching the level of difficulty of the verb."
    "You should not generate sentences that contains only the verb."
    "\n\n"
    "Here you have some examples:"
    ""
    "example_user: Generate a Japanese sentence using the verb 見る in volitive tense."
    "example_assistant: 映画を見に行こう。"
    ""
       ""
    "example_user: Generate a Japanese sentence using the verb 写す in past negative polite tense."
    "example_assistant: 写真を写しませんでした。"
    "" 
    ""
    "example_user: Generate a Japanese sentence using the verb 食べる in conditional tense."
    "example_assistant: 食べたら、元気になります。"
    ""
    
    "\n\n"
    "If any of the tools are unavailable, you can break the fourth wall and tell the user that "
    "they have not implemented them yet and should keep reading to do so.",
)


@tool
def generate_sentences():
    """Generate sentences based on the given verb and tense."""

def tokenize_sentence(sentence: str):
    """Tokenize a sentence into words."""

    tokens =  dango.tokenize(sentence)
    result = []
    for token in tokens:
        result.append({
            "dictionary_form": token.dictionary_form,
            "dictionary_form_reading": token.dictionary_form_reading,
            "part_of_speech": token.part_of_speech.value,
            "surface": token.surface,
            "surface_reading": token.surface_reading,
        })
    return result

def generator_node(state: PhraseGeneratorState) -> PhraseGeneratorState:
    """The generator node. This is where the phrase generator state is manipulated."""
    tool_msg = state.get("messages", [])[-1]
    verbs = state.get("verbs", [])
    tenses = state.get("tenses", [])
    outbound_msgs = []
    generated_sentences = []
    generation_complete = False

    for tool_call in tool_msg.tool_calls:
        if tool_call["name"] == "generate_sentences":
            for verb in verbs:
                for tense in tenses:
                    prompt = f"Generate a Japanese sentence using the verb '{verb}' in {tense} tense."
                    response = llm.invoke([GENERATOR_SYSINT] + [("user", prompt)])
                    generated_sentences.append({
                        "verb": response.verb,
                        "tense": response.tense,
                        "sentence": response.sentence,
                        "tokenized_sentence": tokenize_sentence(response.sentence)
                    })

            if not generated_sentences:
                response = "(no sentence generated)"
            else:
                response = "Generated sentences:\n"
                for sentence in generated_sentences:
                    response += f"- {sentence['verb']} ({sentence['tense']}): {sentence['sentence']}\n\n"
        else:
            raise NotImplementedError(f'Unknown tool call: {tool_call["name"]}')

        print(response)

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
        "finished": generation_complete,
        "tenses": tenses,
        "generated_phrases": generated_sentences
    }

    return state