import ollama
from groq import Groq
import modules.constants as constants
from mistralai import Mistral

def get_client(str):
    if str == 'groq':
        return Groq(api_key=constants.GROQ_API_KEY)
    if str == 'mistral':
        return Mistral(api_key=constants.MISTRAL_API_KEY)

def update_name_list():
    models_list = ollama.list()
    model_names = [model['name'].split(':')[0] for model in models_list['models']]
    model_names = [name for name in model_names if name != 'nomic-embed-text']
    return model_names

def get_models(str):
    client = get_client(str)
    models_list = client.models.list()
    model_names = [model.id for model in models_list.data]
    return model_names

def get_mistral_models():
    client = get_client()
    models_list = client.models.list()
    model_names = [model.id for model in models_list.data]
    return model_names

def update_model_list(check_groq, check_mistral):
    if check_groq:
        return get_models('groq')
    if check_mistral:
        return get_models('mistral')
    else:
        return update_name_list()

def create_custom_model(base_name, new_name, parameter):
    modelfile = f'''
    FROM {base_name}
    SYSTEM {parameter}
    '''
    ollama.create(model=new_name, modelfile=modelfile)
    return update_name_list()