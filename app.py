import ollama
import sys
import fitz
import gradio as gr

# Obtenir la liste des modèles disponibles
def update_name_list():
    models_list = ollama.list()
    model_names = [model['name'].split(':')[0] for model in models_list['models']]
    return model_names

model_names = update_name_list()
conversation_history = []

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
        pdf_text = extract_text_from_pdf(file)
        conversation_history.append({'role': 'system', 'content': pdf_text})
        print(pdf_text)

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

# Launch the app
iface.launch(share=True)
