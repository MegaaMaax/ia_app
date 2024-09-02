import ollama
import sys
import gradio as gr

# Obtenir la liste des modèles
models_list = ollama.list()
model_names = [model['name'].split(':')[0] for model in models_list['models']]

# Fonction pour poser une question à l'IA
def ask_question(question, model):
    stream = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': question}],
        stream=True,
    )
    result = ""
    for chunk in stream:
        result += chunk['message']['content']
    return result

# Créer l'interface utilisateur
iface = gr.Interface(
    fn=ask_question,
    inputs=[
        gr.Textbox(label="Question"),
        gr.Dropdown(
            label="Model",
            choices=model_names,
            value="tinyllama"
        ),
    ],
    outputs="text",
    title="AI Chatbot",
    description="Enter a question to get an answer from the AI chatbot."
)

# Launch the app
iface.launch(server_name="0.0.0.0", server_port=8080)
