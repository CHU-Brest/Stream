import yaml


def load_config(path: str) -> dict:
    """Charge un fichier YAML et retourne son contenu sous forme de dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
