import ollama
import sys
import fitz
import gradio as gr
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Obtenir la liste des modèles disponibles
def update_name_list():
    models_list = ollama.list()
    model_names = [model['name'].split(':')[0] for model in models_list['models']]
    model_names = [name for name in model_names if name != 'nomic-embed-text']
    return model_names

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

# Fonction qui définit la chaîne RAG
def rag_chain_from_pdf(pdf_file, question, model):
    print("debut rag chain")
    retriever = load_and_retrieve_docs_from_pdf(pdf_file)
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
    print("embedding crée")
    stream = ollama.chat(model=model, messages=[{'role': 'user', 'content': formatted_prompt}], stream=True)
    response = ""
    for chunk in stream:
        response += chunk['message']['content']
        yield response

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file.name)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Fonction pour poser une question à l'IA
def ask_question(question, model, file):
    global conversation_history

    if file is not None:
        print("debut rag chain")
        retriever = load_and_retrieve_docs_from_pdf(file)
        retrieved_docs = retriever.invoke(question)
        formatted_context = format_docs(retrieved_docs)
        formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
        print("embedding crée")
        stream = ollama.chat(model=model, messages=[{'role': 'user', 'content': formatted_prompt}], stream=True)
        response = ""
        for chunk in stream:
            response += chunk['message']['content']
            yield response
    else:
        conversation_history.append({'role': 'user', 'content': question})
        print("The model selected is " + model)
        stream = ollama.chat(
            model=model,
            messages=conversation_history,
            stream=True,
        )
        response = ""
        for chunk in stream:
            response += chunk['message']['content']
            yield response
    conversation_history.append({'role': 'assistant', 'content': response})

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

    with gr.Row():
        with gr.Column():
            question = gr.Textbox(label="Question")
            model = gr.Dropdown(label="Model", choices=model_names, value="tinyllama")
            file = gr.File(label="Upload PDF file", file_types=["pdf"])
            submit_button = gr.Button("Submit")
            
            with gr.Accordion("Customize Model", open=False):
                base_model_name = gr.Dropdown(label="Base Model", choices=model_names, value="tinyllama")
                new_model_name = gr.Textbox(label="New Model Name")
                model_parameter = gr.Textbox(label="Model Parameter")
                create_button = gr.Button("Create Model")
            
        with gr.Column():
            output = gr.Textbox(label="Response")

    def handle_create_model(base_name, new_name, parameter):
        new_model_names = create_custom_model(base_name, new_name, parameter)
        return gr.update(choices=new_model_names), gr.update(choices=new_model_names)

    create_button.click(
        fn=handle_create_model,
        inputs=[base_model_name, new_model_name, model_parameter],
        outputs=[model, base_model_name]
    )

    submit_button.click(
        fn=ask_question,
        inputs=[question, model, file],
        outputs=output
    )

# Lancer l'application
iface.launch(server_name="0.0.0.0", server_port=8080)