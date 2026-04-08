from typing import Any


def generate_brest_report(
    row: dict, client: Any, model: str, system_prompt: str
) -> dict:
    """
    Génération spécifique pour Brest.

    Parameters
    ----------
    row : dict
        Une ligne du DataFrame avec au moins 'scenario' et 'generation_id'.
    client : Any
        Client LLM (AnthropicClient, MistralClient, etc.)
    model : str
        Nom du modèle à utiliser.
    system_prompt : str
        Prompt système spécifique au pipeline Brest.

    Returns
    -------
    dict
        La réponse du client LLM avec le texte généré.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["scenario"]},
    ]
    return client.chat(model=model, messages=messages)
