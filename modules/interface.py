import gradio as gr
import ollama
from modules.models import update_model_list, create_custom_model, get_client
from modules.database import upload_database, get_vector_store
from modules.pdf_utils import load_and_retrieve_docs_from_pdf, format_docs
from gradio import ChatMessage

model_names = update_model_list(False)
vector_store = get_vector_store()

def handle_create_model(base_name, new_name, parameter):
    new_model_names = create_custom_model(base_name, new_name, parameter)
    return gr.update(choices=new_model_names), gr.update(choices=new_model_names)

def ask_question(history, question, model, file, check_db, check_groq):
    model_names = update_model_list(check_groq)
    client = get_client()

    if check_db:
        formatted_context = ""
        results = vector_store.similarity_search(query=question, k=5)
        for res in results:
            formatted_context += f"{res.page_content}\n"
        formatted_prompt = f"Question: {question}\n\nContext from database: {formatted_context}"
    else:
        formatted_prompt = f"Question: {question}"

    if file is not None:
        retriever = load_and_retrieve_docs_from_pdf(file)
        retrieved_docs = retriever.invoke(question)
        formatted_context = format_docs(retrieved_docs)
        formatted_prompt += f"\n\nContext from PDF: {formatted_context}"
    
    # Ajouter la question de l'utilisateur à l'historique
    history.append(ChatMessage(role="user", content=question))
    # Ajouter un message vide pour l'assistant à l'historique
    history.append(ChatMessage(role="assistant", content=""))
    yield history

    if check_groq:
        response = ""
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
            stream=True
        )
        for chunk in chat_completion:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
                history[-1].content = response  # Mettre à jour le dernier message de l'assistant
                yield history
    else:
        stream = ollama.chat(model=model, messages=[{'role': 'user', 'content': formatted_prompt}], stream=True)
        response = ""
        for chunk in stream:
            response += chunk['message']['content']
            history[-1].content = response  # Mettre à jour le dernier message de l'assistant
            yield history

def create_interface():
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
                    output = gr.Chatbot(label="Response", type="messages")

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
                inputs=[output, question, model, file, check_db, check_groq],
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

    return iface