# Stream

Génération de comptes rendus d'hospitalisation (CRH) synthétiques à partir de statistiques PMSI nationales et de modèles de langage (LLM).

Stream unifie plusieurs méthodes de génération au sein d'une architecture commune, chacune étant implémentée sous forme de pipeline :

| Pipeline | Source de données | Méthode | Statut |
|----------|-------------------|---------|--------|
| **Brest** | SNDS (Oracle) | Tirage pondéré DP/CCAM/DAS/DMS + LLM | Implémenté |
| **AP-HP** | ATIH (SAS BD) | À définir | En attente |

Projets de référence : [Doppelgänger](https://github.com/CHU-Brest/doppelganger) (CHU Brest) et [Recode-Scenario](https://github.com/24p11/recode-scenario) (AP-HP).

## Architecture

```
Stream/
├── cli.py                        # Point d'entrée CLI
├── runner.py                     # Orchestration du pipeline
├── core/
│   ├── config.py                 # Chargement YAML
│   └── logging.py                # Logger console
├── pipelines/
│   ├── base.py                   # Socle commun (wrappers LLM, get_report, get_client)
│   ├── pipeline_brest.py         # Pipeline Brest (tirage pondéré PMSI)
│   └── pipeline_aphp.py          # Pipeline AP-HP (à implémenter)
└── config/
    ├── prompts.yaml              # System prompts LLM
    └── servers.yaml              # Config pipelines + serveurs LLM
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Éditer `config/servers.yaml` :

```yaml
pipelines:
  brest:
    data:
      input: "data/brest/"       # CSV PMSI extraits via SAS
      output: "reports/brest/"   # CRH générés

servers:
  ollama:
    host: "http://localhost:11434"
    model: "mistral"
  claude:
    api_key: "sk-..."
    model: "claude-sonnet-4-6"
  mistral:
    api_key: "..."
    model: "mistral-large-latest"
```

### Données d'entrée (pipeline Brest)

Déposer les CSV suivants dans le répertoire `data.input` configuré :

| Fichier | Contenu |
|---------|---------|
| `PMSI_DP.csv` | Probabilités P(DP \| GHM5) |
| `PMSI_DAS.csv` | Probabilités P(DAS \| GHM5, AGE, SEXE, DP) |
| `PMSI_CCAM_DP.csv` | Probabilités P(CCAM \| GHM5, DP) |
| `PMSI_DMS.csv` | Durées de séjour (P25, P50, P75) |
| `ALL_CIM10.csv` | Référentiel diagnostics CIM-10 |
| `ALL_CCAM.csv` | Référentiel actes CCAM |
| `ALL_CLASSIF_PMSI.csv` | Référentiel GHM |

Ces fichiers sont produits par les scripts SAS d'extraction (`liabilities/extract_pmsi_tables_SNDS.sas`).

## Utilisation

```bash
# Générer 500 CRH avec Ollama (local)
python cli.py brest --n-sejours 500

# Générer avec Claude, filtré sur les GHM chirurgicaux "06C"
python cli.py brest --client claude --n-sejours 1000 --ghm5 06C

# Générer avec Mistral, 3 actes CCAM max, 4 DAS max
python cli.py brest --client mistral --n-ccam 3 --n-das 4
```

### Options

| Option | Défaut | Description |
|--------|--------|-------------|
| `pipeline` | — | `brest` ou `aphp` |
| `--client` | `ollama` | `ollama`, `claude` ou `mistral` |
| `--n-sejours` | 1000 | Nombre de séjours fictifs |
| `--n-ccam` | 1 | Nombre max d'actes CCAM par séjour |
| `--n-das` | 5 | Nombre max de diagnostics associés |
| `--ghm5` | — | Filtre sur les codes GHM5 (ex. `06C`) |
| `--batch-size` | 1000 | Lignes par fichier Parquet de sortie |

## Sortie

Les CRH sont écrits en fichiers Parquet horodatés dans le répertoire `data.output` :

```
reports/brest/medical_reports_1000_20260405_143022.parquet
```

Schéma : `generation_id`, `scenario`, `report`, `model`, `timestamp`.

## Ajouter un pipeline

Créer une classe héritant de `BasePipeline` et implémenter les 4 méthodes :

```python
from pipelines.base import BasePipeline

class MonPipeline(BasePipeline):
    name = "mon_pipeline"

    def check_data(self) -> None: ...
    def load_data(self) -> dict[str, pl.LazyFrame]: ...
    def get_fictive(self, data, **kwargs) -> pl.DataFrame: ...
    def get_scenario(self, df) -> pl.DataFrame: ...
```

Puis l'enregistrer dans `runner.py` (`PIPELINES`) et `cli.py` (`choices`).
