"""Prompt pour le RAG."""

import logging
import os

from langchain_chroma import Chroma
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from modules.constants import MISTRAL_API_KEY, GROQ_API_KEY
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import FakeEmbeddings

# template du prompt Système
SYSTEM_PROMPT = """
Si tu ne connais pas la réponse, dites simplement que tu ne sais pas. N'essayez pas d'inventer une réponse.
===========
CONTEXTE:
{context}
===========
"""

CHAT = ChatGroq(api_key=GROQ_API_KEY)
DB = Chroma(
        embedding_function=FakeEmbeddings(size=1352),
        persist_directory="db_dir",
    )

def ingest(directory):
    """Fonction principale d'ingestion des documents"""
    # Charge les documents PDF depuis un répertoire
    documents = read_pdfs(directory)
    # Calcule les embeddings et les stocke dans une base vectorielle
    store_embeddings(documents)
    logging.info("Nombre de documents chargés: %i", len(documents))


def read_pdfs(directory):
    """Lit les documents PDF depuis un répertoire"""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Le répertoire {directory} n'existe pas.")
    logging.info("Chargement des fichiers depuis le répertoire %s", directory)
    return PyPDFDirectoryLoader(directory).load_and_split()


def store_embeddings(documents):
    """Stocke les embeddings à partir de text-embedding-ada-002 par défaut"""
    DB.add_documents(documents=documents)


def ask_question(question):
    """Poser une question au modèle."""

    results = DB.as_retriever(search_type="similarity", search_kwargs={
        'k': 10
    }).invoke(input=question)

    # Constitue la séquence de chat avec le conditionnement du bot et la question
    # de l'utilisateur
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        SYSTEM_PROMPT)
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        "QUESTION: {question}")
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    messages = chat_prompt.format_prompt(
        context=results, question=question
    ).to_messages()

    # Pose la question au LLM
    response = CHAT.invoke(input=messages).content

    return response, results


if __name__ == "__main__":
    print("""
Posez simplement votre question:
""")
    print("----------------------------------------------------")
    print("💬 Votre question: ")

    answer, sources = ask_question(input())

    print("----------------------------------------------------")
    print("🤖 Réponse de l'expert: ")
    print(answer)
    print("----------------------------------------------------")
