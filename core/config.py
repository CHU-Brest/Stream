import yaml


def load_config(path: str) -> dict:
    """Charge un fichier YAML et retourne son contenu sous forme de dict."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Fichier de configuration introuvable : {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Erreur de lecture YAML : {e}")
