# Utiliser une image de base Python
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de votre projet dans l'image Docker
COPY . /app

# Installer les dépendances nécessaires
RUN apt -yqq update                     && \
    # nécessaire pour chromadb
    apt -yqq install build-essential    && \
    pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel votre application sera accessible
EXPOSE 8080

# Définir la commande pour démarrer l'application
CMD ["python", "main.py"]