from bottle import route, run, request, static_file
import ollama
import json

# Obtenir la liste des modèles
models_list = ollama.list()
model_names = [model['name'].split(':')[0] for model in models_list['models']]

@route('/')
def index():
    return static_file('index.html', root='.')

@route('/models', method='GET')
def get_models():
    return {'models': model_names}

@route('/ask', method='POST')
def ask_question():
    data = request.json
    question = data.get('question')
    model = data.get('model')
    stream = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': question}],
        stream=True,
    )
    result = ""
    for chunk in stream:
        result += chunk['message']['content']
    return {'answer': result}

# Écouter sur toutes les interfaces réseau, sur le port 8080
run(host='0.0.0.0', port=8080)
