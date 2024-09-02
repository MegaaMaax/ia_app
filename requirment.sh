#!/bin/bash

# Vérifier et installer curl si nécessaire
if ! command -v curl &> /dev/null
then
    echo "curl n'est pas installé. Installation de curl..."
    sudo apt install curl
else
    echo "curl est déjà installé."
fi

# Vérifier et installer pip si nécessaire
if ! command -v pip &> /dev/null
then
    echo "pip n'est pas installé. Installation de pip..."
    sudo apt install pip
else
    echo "pip est déjà installé."
fi

# Installer ollama si nécessaire
if ! command -v ollama &> /dev/null
then
    echo "ollama n'est pas installé. Installation de ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "ollama est déjà installé."
fi

# Installer Flask si nécessaire
if ! pip show Flask &> /dev/null
then
    echo "Flask n'est pas installé. Installation de Flask..."
    pip install Flask
else
    echo "Flask est déjà installé."
fi

# Installer gradio si nécessaire
if ! pip show gradio &> /dev/null
then
    echo "gradio n'est pas installé. Installation de gradio..."
    pip install gradio
else
    echo "gradio est déjà installé."
fi

# Installer ollama si nécessaire
if ! pip show ollama &> /dev/null
then
    echo "ollama n'est pas installé. Installation de ollama..."
    pip install ollama
else
    echo "ollama est déjà installé."
fi

# Installer bottle si nécessaire
if ! pip show bottle &> /dev/null
then
    echo "bottle n'est pas installé. Installation de bottle..."
    pip install bottle
else
    echo "bottle est déjà installé."
fi