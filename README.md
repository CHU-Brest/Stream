# Stream

Génération de comptes rendus d'hospitalisation (CRH) synthétiques à partir de statistiques PMSI nationales et de modèles de langage (LLM).

Stream unifie plusieurs méthodes de génération au sein d'une architecture commune, chacune étant implémentée sous forme de pipeline :

| Pipeline | Source de données | Méthode | Statut |
|----------|-------------------|---------|--------|
| **Brest** | SNDS (Oracle) | Tirage pondéré DP/CCAM/DAS/DMS + LLM | Implémenté |
| **AP-HP** | ATIH (SAS BD) | Tirage pondéré PMSI + LLM | Implémenté |

Projets de référence : [Doppelgänger](https://github.com/CHU-Brest/doppelganger) (CHU Brest) et [Recode-Scenario](https://github.com/24p11/recode-scenario) (AP-HP).

## Architecture

```
Stream/
├── cli.py                        # Point d'entrée CLI
├── runner.py                     # Orchestration du pipeline
├── core/
│   ├── config.py                 # Chargement YAML
│   └── logger.py                 # Logger console
├── pipelines/
│   ├── __init__.py               # Expose les pipelines et modules communs
│   ├── pipeline.py               # Classes de base et clients LLM
│   ├── fictive.py               # Génération de séjours fictifs (Brest/AP-HP)
│   ├── scenario.py              # Transformation en scénarios textuels
│   ├── report.py                # Génération de rapports CRH via LLM
│   ├── brest/
│   │   ├── __init__.py           # Expose BrestPipeline, constants, sampler
│   │   ├── pipeline.py           # Pipeline Brest spécifique
│   │   ├── constants.py          # Constantes et sources de données
│   │   └── sampler.py            # Échantillonnage et génération aléatoire
│   └── aphp/
│       ├── __init__.py           # Expose les modules AP-HP et APHPPipeline
│       ├── pipeline.py           # Pipeline AP-HP spécifique
│       ├── loader.py             # Chargement des référentiels et données PMSI
│       ├── scenario.py           # Construction des scénarios cliniques
│       ├── managment.py          # Classification des types de prise en charge
│       ├── prompt.py             # Génération des prompts utilisateur et système
│       ├── sampler.py            # Échantillonnage et génération aléatoire
│       ├── constants.py          # Constantes et listes de codes
│       ├── referentials/         # Référentiels AP-HP (ICD-10, CCAM, GHM, etc.)
│       └── templates/            # Modèles de prompts système
└── config/
    ├── prompts.yaml              # System prompts LLM
    └── servers.example.yaml      # Template config (copier vers servers.yaml)
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Copier le template et renseigner vos clés API :

```bash
cp config/servers.example.yaml config/servers.yaml
```

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

### Données d'entrée (pipeline AP-HP)

Déposer les fichiers suivants dans le répertoire `data.input` configuré :

| Fichier | Contenu |
|---------|---------|
| `scenarios_*.parquet` | Profils PMSI avec diagnostics principaux et associés |
| `bn_pmsi_related_diag_*.csv` | Diagnostics associés par GHM |
| `bn_pmsi_procedures_*.csv` | Actes CCAM par GHM |

Ces fichiers sont produits par les scripts d'extraction ATIH. Les référentiels AP-HP doivent être placés dans `data.referentials`.

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
| `--n-ccam` | 1 | Nombre max d'actes CCAM par séjour (Brest uniquement) |
| `--n-das` | 5 | Nombre max de diagnostics associés (Brest uniquement) |
| `--ghm5` | — | Filtre sur les codes GHM5 (ex. `06C`) |
| `--batch-size` | 1000 | Lignes par fichier Parquet de sortie |

## Sortie

Les CRH sont écrits en fichiers Parquet horodatés dans le répertoire `data.output` :

```
reports/brest/medical_reports_1000_20260405_143022.parquet
```

Schéma : `generation_id`, `scenario`, `report`, `model`, `timestamp`.

## Tests

Pour exécuter les tests unitaires, utilisez la commande suivante :

```bash
python -m pytest tests/ -v
```

Les tests couvrent :
- L'initialisation des pipelines Brest et AP-HP
- La vérification des données d'entrée
- L'intégration des pipelines avec le runner

## Architecture des Pipelines

Les pipelines suivent une architecture modulaire avec trois étapes principales :

```
get_fictive() → get_scenario() → get_report()
```

1. **get_fictive()** : Génère des séjours fictifs à partir des données chargées.
2. **get_scenario()** : Transforme les séjours fictifs en scénarios textuels pour le LLM.
3. **get_report()** : Génère les comptes rendus d'hospitalisation (CRH) à partir des scénarios.

## Modules communs

Trois nouveaux modules ont été ajoutés pour centraliser la logique commune entre les pipelines :

### `pipelines/fictive.py`

Module responsable de la génération de séjours fictifs à partir des données PMSI.

**Fonction principale** :
- `generate_fictive_stays(data, n_sejours, pipeline_type, **kwargs)`

**Responsabilités** :
- Génération de séjours fictifs pour les pipelines Brest et AP-HP
- Logique d'échantillonnage spécifique à chaque pipeline
- Réduction de la duplication de code

### `pipelines/scenario.py`

Module responsable de la transformation des séjours fictifs en scénarios textuels pour les LLM.

**Fonction principale** :
- `format_scenarios(df, pipeline_type, **kwargs)`

**Responsabilités** :
- Transformation des séjours en prompts textuels
- Formatage spécifique Brest (simple) et AP-HP (riche)
- Gestion des templates et règles ATIH

### `pipelines/report.py`

Module responsable de la génération de rapports CRH via appels LLM.

**Fonction principale** :
- `generate_reports(df, client, model, batch_size, output_dir, pipeline_type)`

**Responsabilités** :
- Interaction avec les clients LLM (Anthropic, Mistral, Ollama)
- Gestion des batches et persistance des résultats
- Formatage des réponses LLM

## Modules communs

Trois nouveaux modules ont été ajoutés pour centraliser la logique commune entre les pipelines :

### `pipelines/fictive.py`

Module responsable de la génération de séjours fictifs à partir des données PMSI.

**Fonction principale** :
- `generate_fictive_stays(data, n_sejours, pipeline_type, **kwargs)`

**Responsabilités** :
- Génération de séjours fictifs pour les pipelines Brest et AP-HP
- Logique d'échantillonnage spécifique à chaque pipeline
- Réduction de la duplication de code

### `pipelines/scenario.py`

Module responsable de la transformation des séjours fictifs en scénarios textuels pour les LLM.

**Fonction principale** :
- `format_scenarios(df, pipeline_type, **kwargs)`

**Responsabilités** :
- Transformation des séjours en prompts textuels
- Formatage spécifique Brest (simple) et AP-HP (riche)
- Gestion des templates et règles ATIH

### `pipelines/report.py`

Module responsable de la génération de rapports CRH via appels LLM.

**Fonction principale** :
- `generate_reports(df, client, model, batch_size, output_dir, pipeline_type)`

**Responsabilités** :
- Interaction avec les clients LLM (Anthropic, Mistral, Ollama)
- Gestion des batches et persistance des résultats
- Formatage des réponses LLM

## Comparaison des Pipelines

| Aspect | Pipeline Brest (CHU Brest) | Pipeline AP-HP (Paris) |
|--------|----------------------------|-------------------------|
| **Source de données** | SNDS (Oracle) | ATIH (SAS BD) |
| **Méthode de tirage** | Pondéré DP/CCAM/DAS/DMS | Pondéré PMSI + règles ATIH |
| **Modules communs** | fictive.py, scenario.py | fictive.py, scenario.py, report.py |
| **Logique métier** | Simple (CHU Brest) | Complexe (référentiels ATIH) |
| **Classification** | Basique | MCO/SSR/HAD (managment.py) |
| **Validation** | Standard | Règles ATIH spécifiques |
| **Templates** | Génériques | Spécifiques AP-HP |
| **Sur-couche** | Non | Oui (prompts enrichis) |

## Intégration dans les pipelines

Les pipelines Brest et AP-HP ont été mis à jour pour utiliser ces modules communs :

### Brest Pipeline
```python
class BrestPipeline(BasePipeline):
    def get_fictive(self, data, **kwargs):
        return generate_fictive_stays(data, pipeline_type="brest", **kwargs)
    
    def get_scenario(self, df):
        return format_scenarios(df, pipeline_type="brest")
```

### AP-HP Pipeline
Le pipeline AP-HP utilise les modules communs avec une sur-couche spécifique pour la génération de rapports. Cette sur-couche implémente la logique métier AP-HP pour :

- **Gestion des référentiels ATIH** : Chargement et validation des codes ICD-10, CCAM et GHM spécifiques à l'AP-HP
- **Classification des prises en charge** : Application des règles de managment (MCO, SSR, HAD) selon les référentiels ATIH
- **Génération de prompts enrichis** : Intégration des templates système spécifiques AP-HP avec règles de formatage ATIH
- **Validation des scénarios** : Vérification de la cohérence clinique selon les règles métiers AP-HP

```python
class APHPPipeline(BasePipeline):
    def get_fictive(self, data, **kwargs):
        return generate_fictive_stays(data, pipeline_type="aphp", **kwargs)
    
    def get_scenario(self, df):
        return format_scenarios(df, pipeline_type="aphp", **kwargs)
    
    def get_report(self, df, client, model, **kwargs):
        return generate_reports(df, client, model, pipeline_type="aphp", **kwargs)
```

## Ajouter un pipeline

Créer une classe héritant de `BasePipeline` et implémenter les 4 méthodes :

```python
from pipelines.pipeline import BasePipeline

class MonPipeline(BasePipeline):
    name = "mon_pipeline"

    def check_data(self) -> None:
        """Verify that source data is present and prepare it if needed."""
        # Ajoutez des messages de log pour faciliter le débogage
        self.logger.info("Vérification des données d'entrée pour le pipeline %s.", self.name)
        # Votre logique de vérification des données ici
        self.logger.info("Données vérifiées avec succès.")

    def load_data(self) -> dict[str, pl.LazyFrame]:
        """Load prepared data as LazyFrames."""
        # Ajoutez des messages de log pour suivre le chargement des données
        self.logger.info("Chargement des données pour le pipeline %s.", self.name)
        # Votre logique de chargement des données ici
        self.logger.info("Données chargées avec succès.")
        return {}

    def get_fictive(self, data, **kwargs) -> pl.DataFrame:
        """Generate fictitious hospital stays from loaded data."""
        # Ajoutez des messages de log pour suivre la génération des séjours fictifs
        self.logger.info("Génération de %d séjours fictifs.", kwargs.get("n_sejours", 1000))
        # Votre logique de génération des séjours fictifs ici
        self.logger.info("Génération des séjours fictifs terminée avec succès.")
        return pl.DataFrame()

    def get_scenario(self, df) -> pl.DataFrame:
        """Transform fictitious stays into text scenarios for the LLM."""
        # Ajoutez des messages de log pour suivre le formatage des scénarios
        self.logger.info("Formatage des scénarios pour %d séjours.", len(df))
        # Votre logique de formatage des scénarios ici
        self.logger.info("Formatage des scénarios terminé avec succès.")
        return df
```

Puis l'enregistrer dans `runner.py` (`PIPELINES`) et `cli.py` (`choices`).
        self.logger.info("Données vérifiées avec succès.")

    def load_data(self) -> dict[str, pl.LazyFrame]:
        """Load prepared data as LazyFrames."""
        # Ajoutez des messages de log pour suivre le chargement des données
        self.logger.info("Chargement des données pour le pipeline %s.", self.name)
        # Votre logique de chargement des données ici
        self.logger.info("Données chargées avec succès.")
        return {}

    def get_fictive(self, data, **kwargs) -> pl.DataFrame:
        """Generate fictitious hospital stays from loaded data."""
        # Ajoutez des messages de log pour suivre la génération des séjours fictifs
        self.logger.info("Génération de %d séjours fictifs.", kwargs.get("n_sejours", 1000))
        # Votre logique de génération des séjours fictifs ici
        self.logger.info("Génération des séjours fictifs terminée avec succès.")
        return pl.DataFrame()

    def get_scenario(self, df) -> pl.DataFrame:
        """Transform fictitious stays into text scenarios for the LLM."""
        # Ajoutez des messages de log pour suivre le formatage des scénarios
        self.logger.info("Formatage des scénarios pour %d séjours.", len(df))
        # Votre logique de formatage des scénarios ici
        self.logger.info("Formatage des scénarios terminé avec succès.")
        return df
```

Puis l'enregistrer dans `runner.py` (`PIPELINES`) et `cli.py` (`choices`).
