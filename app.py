from flask import Flask, send_file
import gradio as gr
import ollama

app = Flask(__name__)

# Obtenir la liste des modèles
models_list = ollama.list()
model_names = [model['name'].split(':')[0] for model in models_list['models']]

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

iface = gr.Interface(
    fn=ask_question,
    inputs=[
        gr.Textbox(label="Question"),
        gr.Dropdown(
            label="Model",
            choices=model_names,
            value="dolphin-mistral"
        ),
    ],
    outputs="text",
    title="AI Chatbot",
    description="Enter a question to get an answer from the AI chatbot."
)

@app.route('/')
def index():
    return send_file(iface.launch())

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)