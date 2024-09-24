import ollama
import os
import fitz
import gradio as gr
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
import constants

vector_store = Chroma(
    collection_name="rag_db",
    embedding_function=OllamaEmbeddings(model='nomic-embed-text'),
    persist_directory="db_dir",
)

client = Groq(
    api_key=constants.GROQ_API_KEY,
)

def upload_database(file):
    doc = fitz.open(file.name)
    text = ""
    for page in doc:
        text += page.get_text()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(text)
    vector_store.add_texts(texts=splits)
    return "Database uploaded successfully"

# Obtenir la liste des modèles disponibles
def update_name_list():
    models_list = ollama.list()
    model_names = [model['name'].split(':')[0] for model in models_list['models']]
    model_names = [name for name in model_names if name != 'nomic-embed-text']
    return model_names

def get_groq_models():
    models_list = client.models.list()
    model_names = [model.id for model in models_list.data]
    return model_names

def update_model_list(check_groq):
    if check_groq:
        return get_groq_models()
    else:
        return update_name_list()

model_names = update_name_list()
conversation_history = []

# Fonction pour charger et diviser les documents PDF
def load_and_retrieve_docs_from_pdf(pdf_file):
    doc = fitz.open(pdf_file.name)
    text = ""
    for page in doc:
        text += page.get_text()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(text)
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vectorstore = Chroma.from_texts(texts=splits, embedding=embeddings)
    return vectorstore.as_retriever()

# Fonction pour formater les documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file.name)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Fonction pour poser une question à l'IA
def ask_question(question, model, file, check_db, check_groq):
    global conversation_history

    # Mettre à jour la liste des modèles
    model_names = update_model_list(check_groq)

    if check_db:
        print("Début search in db")
        formatted_context = ""
        results = vector_store.similarity_search(query=question, k=5)
        for res in results:
            formatted_context += f"{res.page_content}\n"
        formatted_prompt = f"Question: {question}\n\nContext from database: {formatted_context}"
        print("Recherche dans la base de données effectuée")
    else:
        formatted_prompt = f"Question: {question}"

    if file is not None:
        print("Début RAG chain")
        retriever = load_and_retrieve_docs_from_pdf(file)
        retrieved_docs = retriever.invoke(question)
        formatted_context = format_docs(retrieved_docs)
        formatted_prompt += f"\n\nContext from PDF: {formatted_context}"
        print("Embedding créé")
    
    if check_groq:
        print("Groq activated")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un assistant, mais quand tu ne connais pas une réponse tu dois répondre: Je ne sais pas. Sans oublier de répondre en français"
                },
                {
                    "role": "user",
                    "content": formatted_prompt,
                }
            ],
            model=model,
        )
        response = chat_completion.choices[0].message.content
        yield response
    else:
        stream = ollama.chat(model=model, messages=[{'role': 'user', 'content': formatted_prompt}], stream=True)
        response = ""
        for chunk in stream:
            response += chunk['message']['content']
            yield response

    conversation_history.append({'role': 'user', 'content': question})
    conversation_history.append({'role': 'assistant', 'content': response})

def handle_create_model(base_name, new_name, parameter):
    new_model_names = create_custom_model(base_name, new_name, parameter)
    return gr.update(choices=new_model_names), gr.update(choices=new_model_names)

# Fonction pour créer un modèle personnalisé
def create_custom_model(base_name, new_name, parameter):
    modelfile = f'''
    FROM {base_name}
    SYSTEM {parameter}
    '''
    ollama.create(model=new_name, modelfile=modelfile)
    return update_name_list()

# Créer l'interface utilisateur
with gr.Blocks(theme='gradio/soft') as iface:
    gr.Markdown("# AI Chatbot")

    with gr.Tab("Chatbot"):
        with gr.Row():
            with gr.Column():
                question = gr.Textbox(label="Question")
                model = gr.Dropdown(label="Model", choices=model_names, value="tinyllama")
                check_db = gr.Checkbox(label="Use Database")
                check_groq = gr.Checkbox(label="Use Groq")
                file = gr.File(label="Upload PDF file", file_types=["pdf"])
                submit_button = gr.Button("Submit")
                
            with gr.Column():
                output = gr.Textbox(label="Response")

        def update_model_dropdown(check_groq):
            model_names = update_model_list(check_groq)
            return gr.update(choices=model_names)

        check_groq.change(
            fn=update_model_dropdown,
            inputs=[check_groq],
            outputs=[model]
        )

        submit_button.click(
            fn=ask_question,
            inputs=[question, model, file, check_db, check_groq],
            outputs=output
        )

    with gr.Tab("Customize Model"):
        base_model_name = gr.Dropdown(label="Base Model", choices=model_names, value="tinyllama")
        new_model_name = gr.Textbox(label="New Model Name")
        model_parameter = gr.Textbox(label="Model Parameter")
        create_button = gr.Button("Create Model")

        create_button.click(
            fn=handle_create_model,
            inputs=[base_model_name, new_model_name, model_parameter],
            outputs=[model, base_model_name]
        )

    with gr.Tab("Upload database"):
        with gr.Row():
            with gr.Column():
                db_file = gr.File(label="Upload Database file", file_types=["pdf"])
                submit_db_button = gr.Button("Submit")

            with gr.Column():
                db_output = gr.Textbox(label="Response")

        submit_db_button.click(
            fn=upload_database,
            inputs=[db_file],
            outputs=db_output
        )

# Lancer l'application
iface.launch()

#server_name="0.0.0.0", server_port=8080