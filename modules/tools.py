from unittest.mock import Base
import requests
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from langchain.tools import tool

from modules.prompt_rag import ask_question

# -----------------------------------------------
# Approche decorator
# -----------------------------------------------


@tool
def get_pokemon_details(pokemon_name):
    """Get pokemon details by its pokemon name.

    Args:
        pokemon_name (str): The name of the pokemon to get details for.

    Returns:
        dict: A dictionary containing the details of the pokemon."""
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
    response = requests.get(url, timeout=15)
    if response.status_code == 200:
        data = response.json()
        return {
            "name": data["name"],
            "height": data["height"],
            "weight": data["weight"],
            "base_experience": data["base_experience"],
            "abilities": [ability["ability"]["name"] for ability in data["abilities"]],
        }
    return None


@tool
def get_pokemon_locations(pokemon_name):
    """Get the locations of a pokemon.

    Args:
        pokemon_name (str): The name of the pokemon to get locations for.

    Returns:
        list: A list of locations of the pokemon."""
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}/encounters"
    response = requests.get(url, timeout=15)

    if response.status_code == 200:
        data = response.json()
        return [location["location_area"]["name"] for location in data]
    return None


# -----------------------------------------------
# Approche structured tools
# -----------------------------------------------


class LyricsInput(BaseModel):
    """Input for the lyrics tool."""

    artist: str = Field(description="should be an artist name")
    title: str = Field(description="should be the title of their song")


def get_lyrics(artist: str, title: str):
    """Gets the lyrics of a song."""
    url = f"https://api.lyrics.ovh/v1/{artist}/{title}"
    response = requests.get(url, timeout=15)

    if response.status_code == 200:
        return response.json()["lyrics"]
    return None


if __name__ == "__main__":
    print(get_lyrics("Rage Against The Machine", "Killing In The Name"))


class RAG(BaseModel):
    """Input for the RAG tool."""

    question: str = Field(description="The question to ask the RAG model")

def get_rag_response(question: str):
    """Get a response from the RAG model."""
    return ask_question(question)
