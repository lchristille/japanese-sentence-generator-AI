from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.constants import START
from langgraph.graph import StateGraph

load_dotenv()

from agents.chatbot import make_chatbot, PhraseGeneratorState
from agents.sentence_expert import make_sentence_expert


chatbot = make_chatbot(ChatGoogleGenerativeAI(model="gemini-2.0-flash"))
sentence_expert = make_sentence_expert(ChatGoogleGenerativeAI(model="gemini-2.0-flash"))

builder = StateGraph(PhraseGeneratorState)
builder.add_node("chatbot", chatbot)
builder.add_node("sentence_expert", sentence_expert)
builder.add_edge(START, "chatbot")
graph = builder.compile()

config = {"recursion_limit": 100}

state = graph.invoke({"messages": []}, config)