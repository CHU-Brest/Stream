from typing import Any


def generate_aphp_report(
    row: dict,
    client: Any,
    model: str,
) -> dict:
    """
    Génération spécifique pour AP-HP.

    Parameters
    ----------
    row : dict
        Une ligne du DataFrame avec au moins 'scenario' et 'generation_id'.
        Doit contenir 'system_prompt' si tu veux l'overrider par ligne.
    client : Any
        Client LLM (AnthropicClient, MistralClient, etc.)
    model : str
        Nom du modèle à utiliser.
    system_prompt : str
        Prompt système spécifique au pipeline AP-HP.

    Returns
    -------
    dict
        La réponse du client LLM avec le texte généré.
    """
    messages = [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": row["scenario"]},
    ]
    return client.chat(model=model, messages=messages)
