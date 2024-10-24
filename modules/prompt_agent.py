from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import StructuredTool, Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_groq import ChatGroq
from gradio import ChatMessage

from modules.constants import GROQ_API_KEY
import modules.tools as tools

# Instanciation du chat
_CHAT = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-70b-versatile")

# Déclaration des outils
_TOOLS = [
    Tool.from_function(
        func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
        name="Wikipedia",
        description="Only useful when you need an answer about the recent news or about famous people. Don't use it if you can answer the question yourself",
        handle_tool_error=True,
    ),
    Tool.from_function(
        func=tools.get_pokemon_details,
        name="PokemonDetails",
        description="Useful when you need to answer about pokemon characteristics",
        handle_tool_error=True,
    ),
    Tool.from_function(
        func=tools.get_pokemon_locations,
        name="PokemonLocations",
        description="Useful when you need to answer about pokemon locations",
        handle_tool_error=True,
    ),
    StructuredTool.from_function(
        func=tools.get_rag_response,
        name="RAG",
        description="Only useful when you need to answer about a school project or about Epitech. Don't use it if you don't know the question or if you can answer the question yourself",
        handle_tool_error=True,
        args_schema=tools.RAG,
    ),
    StructuredTool.from_function(
        func=tools.get_lyrics,
        name="Lyrics",
        description="Useful when you need to get lyrics from an artist and title",
        handle_tool_error=True,
        args_schema=tools.LyricsInput,
    ),
]

# Initialisation de l'agent
prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_tool_calling_agent(_CHAT, _TOOLS, prompt)
agent_executor = AgentExecutor(agent=agent, tools=_TOOLS, verbose=True)

def ask_question(chat_input, history):
    """Poser une question au modèle avec l'agent et ses outils."""
    response = None
    used_tools = []
    question = chat_input["text"]

    print("Question:", question)
    
    # Formatage de la question dans history
    history.append(ChatMessage(role="user", content=question))
    history.append(ChatMessage(role="assistant", content=""))
    yield history

    stream_iterator = agent_executor.stream({"input": question})
    # Lance l'évaluation du prompt
    for chunk in stream_iterator:
        # Enregistrement des actions du LLM
        if "actions" in chunk:
            for action in chunk["actions"]:
                used_tools.append(action.tool)
                response = action.tool_input
        if "output" in chunk:
            response = chunk["output"]
            history[-1].content = response
            yield history

    return history

if __name__ == "__main__":
    # Test de la fonction ask_question
    chat_input = {"text": "Quelle est la capitale de la France ?"}
    history = []
    for h in ask_question(chat_input, history):
        print(h[-1].content)