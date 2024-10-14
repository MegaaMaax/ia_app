import os
import gradio as gr
import ollama
from modules.models import update_model_list, create_custom_model, get_client
from modules.database import upload_database, get_vector_store
from modules.pdf_utils import load_and_retrieve_docs_from_pdf, format_docs, encode_image_base64
from modules.sql_query import sql_question
from gradio import ChatMessage

model_names = update_model_list(False, False)
vector_store = get_vector_store()

def handle_create_model(base_name, new_name, parameter):
    new_model_names = create_custom_model(base_name, new_name, parameter)
    return gr.update(choices=new_model_names), gr.update(choices=new_model_names)

def ask_question(history, chat_input, model, check_db, check_groq, check_mistral):
    model_names = update_model_list(check_groq, check_mistral)
    question = chat_input["text"]
    if check_db:
        formatted_context = ""
        results = vector_store.similarity_search(query=question, k=5)
        for res in results:
            formatted_context += f"{res.page_content}\n"
        formatted_prompt = f"Question: {question}\n\nContext from database: {formatted_context}"
    else:
        formatted_prompt = f"Question: {question}"

    if chat_input["files"] and len(chat_input["files"]) > 0:
        file_path = chat_input["files"][0]
        file_name, file_extension = os.path.splitext(file_path)
        if file_extension.lower() == ".pdf":
            retriever = load_and_retrieve_docs_from_pdf(file_path)
            retrieved_docs = retriever.invoke(question)
            formatted_context = format_docs(retrieved_docs)
            formatted_prompt += f"\n\nContext from PDF: {formatted_context}"
    
    history.append(ChatMessage(role="user", content=question))
    history.append(ChatMessage(role="assistant", content=""))
    yield history

    if check_groq:
        client = get_client("groq")
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
                history[-1].content = response
                yield history
    elif check_mistral:
        client = get_client("mistral")
        response = ""
        if "pixtral" in model and chat_input["files"] and len(chat_input["files"]) > 0:
            list_image = encode_image_base64(chat_input["files"][0])
            chat_completion = client.chat.stream(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": formatted_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{list_image}"
                            }
                        ]
                    }
                ],
                model=model
            )
        else:
            chat_completion = client.chat.stream(
                messages=[
                    {
                        "role": "user",
                        "content": formatted_prompt,
                    }
                ],
                model=model
            )
        for chunk in chat_completion:
            if chunk.data.choices[0].delta.content is not None:
                response += chunk.data.choices[0].delta.content
                history[-1].content = response
                yield history
    else:
        stream = ollama.chat(model=model, messages=[{'role': 'user', 'content': formatted_prompt}], stream=True)
        response = ""
        for chunk in stream:
            response += chunk['message']['content']
            history[-1].content = response
            yield history

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as iface:
        with gr.Tab("Chatbot"):
            output = gr.Chatbot(show_label=False, type="messages")
            chat_input = gr.MultimodalTextbox(show_label=False, placeholder="Entrée votre question ici", file_count="single")
            with gr.Accordion(label="Paramètres avancés", open=False):
                model = gr.Dropdown(label="Model", choices=model_names, value="tinyllama")
                check_db = gr.Checkbox(label="Use Database")
                check_groq = gr.Checkbox(label="Use Groq")
                check_mistral = gr.Checkbox(label="Use Mistral")
            clear_button = gr.ClearButton([chat_input, output])

            def update_model_name(check_groq, check_mistral):
                model_names = update_model_list(check_groq, check_mistral)
                return gr.update(choices=model_names)
            
            chat_input.submit(
                fn=ask_question,
                inputs=[output, chat_input, model, check_db, check_groq, check_mistral],
                outputs=output
            )

            check_groq.change(
                fn=update_model_name,
                inputs=[check_groq, check_mistral],
                outputs=[model]
            )

            check_mistral.change(
                fn=update_model_name,
                inputs=[check_groq, check_mistral],
                outputs=[model]
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
                    db_file = gr.File(label="Upload Database file", file_types=["file"])
                    submit_db_button = gr.Button("Submit")

                with gr.Column():
                    db_output = gr.Textbox(label="Response")

            submit_db_button.click(
                fn=upload_database,
                inputs=[db_file],
                outputs=db_output
            )

        with gr.Tab("SQL Database"):
            gr.Image("https://i.ibb.co/Mn3tNmS/db-schema.png", show_label=False)
            with gr.Row():
                with gr.Column():
                    question = gr.Textbox(label="Question")
                    submit_sql_button = gr.Button("Submit")
                with gr.Column():
                    db_output = gr.Textbox(label="Response")

            submit_sql_button.click(
                fn=sql_question,
                inputs=[question],
                outputs=db_output
            )

            question.submit(
                fn=sql_question,
                inputs=[question],
                outputs=db_output
            )

    return iface
