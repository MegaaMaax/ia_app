from bottle import route, run

@route('/')
def index():
    return 'Hello World!'

# Écouter sur toutes les interfaces réseau, sur le port 8080
run(host='0.0.0.0', port=8080)
