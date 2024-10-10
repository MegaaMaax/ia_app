import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

# Charge les fichiers .env
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
"""Clé d'API Groq"""

template = """Based on the table schema below, write a SQL query and only the SQL query nothing else without forgetting the semicolon, that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)


db_url = "mysql+mysqlconnector://root:Maxime.74@localhost:3306/Chinook"
db = SQLDatabase.from_uri(db_url)

# Initialiser Groq avec la clé d'API
groq = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-70b-versatile")


def get_schema(_):
    return db.get_table_info()

def run_query(query):
    return db.run(query)

def print_query(query):
    print("Generated SQL Query:", query)
    return query

sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | groq.bind(stop=["\nSQL Result:"])
    | StrOutputParser()
    | print_query
)

template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_template(template)

def run_query(query):
    return db.run(query)

def print_prompt_response(response):
    print("Generated Response:", response)
    return response

full_chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
        schema=get_schema,
        response=lambda vars: run_query(vars["query"]),
    )
    | prompt_response
    | groq
    | print_prompt_response
)

def sql_question(question):
    response = full_chain.invoke({"question": question})
    return response.content