import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from azure.identity import DefaultAzureCredential
from langchain_community.agent_toolkits import PowerBIToolkit, create_pbi_agent
from langchain_community.utilities.powerbi import PowerBIDataset
from modules.constants import GROQ_API_KEY

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-70b-versatile",
)

toolkit = PowerBIToolkit(
    powerbi=PowerBIDataset(
        dataset_id="d8951070-5d89-40c5-bf3a-2e36575a030e",
        table_names=["table1", "table2"],
        credential=DefaultAzureCredential(),
    ),
    llm=llm,
)

agent_executor = create_pbi_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
)

def ask_bi_question(question):
    return agent_executor.run(question)
