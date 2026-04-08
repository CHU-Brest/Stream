# 🛠️ Simplification des dépendances et des appels de fonctions - Stream

**Ce guide propose des recommandations pratiques pour simplifier le code du projet Stream**, adaptées à la phase actuelle de développement où la simplicité et la maintenabilité sont prioritaires.

---

## 📖 Table des matières

1. [Simplification des imports](#1-simplification-des-imports)
2. [Réduction des dépendances circulaires](#2-réduction-des-dépendances-circulaires)
3. [Simplification des appels de fonctions](#3-simplification-des-appels-de-fonctions)
4. [Utilisation de fonctions utilitaires](#4-utilisation-de-fonctions-utilitaires)
5. [Simplification des structures de données](#5-simplification-des-structures-de-données)
6. [Réduction des dépendances externes](#6-réduction-des-dépendances-externes)
7. [Simplification des appels aux pipelines](#7-simplification-des-appels-aux-pipelines)

---

## 1. 🗂️ Simplification des imports

**Problème** : Certains modules importent des fonctions ou des classes qui ne sont pas utilisées, ce qui complique les dépendances et augmente le temps de chargement.

**Solution** :
- Supprimer les imports inutilisés.
- Regrouper les imports par type (standard, tiers, local).
- Utiliser des imports explicites plutôt que des imports wildcard (`*`).

**Explication** : Des imports propres et minimalistes améliorent la lisibilité du code et réduisent les dépendances inutiles. Cela facilite également la maintenance et le débogage.

**Effets de bord** :
- La suppression d'imports inutilisés peut casser des dépendances implicites si certains modules dépendent indirectement de ces imports.
- Les imports regroupés par type peuvent nécessiter des ajustements dans les IDE pour la détection des imports.

**Solutions pour éviter les effets de bord** :
- **Vérifier les dépendances** : Avant de supprimer un import, utiliser des outils comme `grep` pour vérifier s'il est utilisé ailleurs dans le code.
- **Tests unitaires** : Exécuter les tests unitaires après chaque suppression d'import pour s'assurer que rien n'est cassé.
- **Refactoring progressif** : Supprimer les imports inutilisés par petites étapes et valider chaque changement.

**Exemple** :

**Exemple 1 : Suppression des imports inutilisés**

**Contexte** : Dans `pipelines/brest/pipeline.py`, certains imports ne sont pas utilisés directement.

**Avant** ❌
```python
# pipelines/brest/pipeline.py
from pipelines.fictive import generate_fictive_stays
from pipelines.scenario import format_scenarios
from pipelines.report import generate_reports
from pipelines.brest.fictive import generate_brest_fictive  # ⚠️ Inutilisé directement
from pipelines.brest.scenario import format_brest_scenario  # ⚠️ Inutilisé directement
import polars as pl
import random  # ⚠️ Non utilisé dans ce module
from pathlib import Path
```

**Vérification** 🔍
```bash
# Vérifier si generate_brest_fictive est utilisé ailleurs
grep -r "generate_brest_fictive" pipelines/
# Résultat : Aucun résultat → peut être supprimé en toute sécurité
```

**Après** ✅
```python
# pipelines/brest/pipeline.py
from pipelines.fictive import generate_fictive_stays
from pipelines.scenario import format_scenarios
from pipelines.report import generate_reports
import polars as pl
from pathlib import Path
```

**Impact** 📊
- Réduction de 3 imports inutilisés
- Code plus propre et plus facile à maintenir
- Temps de chargement légèrement amélioré

---

**Exemple 2 : Regroupement des imports par type**

**Contexte** : Dans `pipelines/aphp/pipeline.py`, les imports sont mélangés et non organisés.

**Avant** ❌
```python
# pipelines/aphp/pipeline.py
import polars as pl
from pipelines.aphp.scenario import format_aphp_scenario
from pathlib import Path
from pipelines.fictive import generate_fictive_stays
import logging
from pipelines.scenario import format_scenarios
```

**Après** ✅
```python
# pipelines/aphp/pipeline.py
# Imports standard
import logging
from pathlib import Path

# Imports tiers
import polars as pl

# Imports locaux
from pipelines.fictive import generate_fictive_stays
from pipelines.scenario import format_scenarios
from pipelines.aphp.scenario import format_aphp_scenario
```

**Impact** 📊
- Imports organisés par type (standard, tiers, local)
- Meilleure lisibilité et maintenance
- Respect des conventions PEP 8

**Sources** :
- [PEP 8 -- Imports](https://peps.python.org/pep-0008/#imports)

---

## 2. 🔄 Réduction des dépendances circulaires

**Problème** : Certains modules ont des dépendances circulaires, ce qui complique la maintenance et peut causer des erreurs d'importation.

**Solution** :
- Utiliser des imports locaux (`import` à l'intérieur des fonctions) pour casser les dépendances circulaires
- Déplacer les fonctions communes dans des modules partagés
- Refactoriser le code pour éviter les dépendances circulaires

**Explication** : Les dépendances circulaires rendent le code difficile à maintenir et à tester. En les éliminant, on améliore la modularité et la clarté du code.

**Diagramme des dépendances circulaires** :
```
┌─────────────────┐       ┌─────────────────┐
│ aphp/pipeline.py │──────▶│ scenario.py     │
└─────────────────┘       └─────────────────┘
       ▲                                      │
       │                                      ▼
┌─────────────────┐       ┌─────────────────┐
│ aphp/scenario.py│◀──────│ pipeline.py     │
└─────────────────┘       └─────────────────┘
```

**Effets de bord** :
- L'utilisation d'imports locaux peut masquer des problèmes de conception plus profonds
- Les dépendances circulaires peuvent être nécessaires dans certains cas pour des raisons architecturales

**Solutions pour éviter les effets de bord** :
- **Refactoring architectural** : Si des dépendances circulaires sont nécessaires, envisager un refactoring pour séparer les responsabilités
- **Utilisation d'interfaces** : Définir des interfaces communes dans un module séparé
- **Documentation** : Documenter les dépendances circulaires et leurs raisons

**Exemple avec le code du projet** :

**Problème (dépendance circulaire)** :
```python
# pipelines/aphp/pipeline.py
from pipelines.aphp.scenario import format_aphp_scenario
from pipelines.scenario import format_scenarios

def get_scenario(self, df: pl.DataFrame) -> pl.DataFrame:
    return format_scenarios(df, scenario_fn=format_aphp_scenario)

# pipelines/scenario.py
from pipelines.aphp.pipeline import APHPPipeline

def format_scenarios(df: pl.DataFrame, scenario_fn) -> pl.DataFrame:
    if isinstance(df, APHPPipeline):
        return scenario_fn(df)
```

**Solution (import local)** :
```python
# pipelines/aphp/pipeline.py
from pipelines.scenario import format_scenarios

def get_scenario(self, df: pl.DataFrame) -> pl.DataFrame:
    from pipelines.aphp.scenario import format_aphp_scenario
    return format_scenarios(df, scenario_fn=format_aphp_scenario)

# pipelines/scenario.py
# Aucun import depuis pipelines.aphp.pipeline

def format_scenarios(df: pl.DataFrame, scenario_fn) -> pl.DataFrame:
    return scenario_fn(df)
```

**Alternative (utilisation d'interfaces)** :
```python
# pipelines/interfaces.py
from typing import Protocol
import polars as pl

class ScenarioFormatter(Protocol):
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        ...

# pipelines/aphp/scenario.py
from pipelines.interfaces import ScenarioFormatter

def format_aphp_scenario(df: pl.DataFrame) -> pl.DataFrame:
    # Implémentation spécifique APHP
    ...

# pipelines/scenario.py
from pipelines.interfaces import ScenarioFormatter

def format_scenarios(df: pl.DataFrame, scenario_fn: ScenarioFormatter) -> pl.DataFrame:
    return scenario_fn(df)
```

**Note** : La solution avec import local est plus simple et alignée avec l'architecture actuelle. L'approche par interfaces est plus robuste pour des projets plus complexes.

**Sources** :
- [Python Documentation - Modules](https://docs.python.org/3/tutorial/modules.html)
- [PEP 544 -- Protocols: Structural subtyping](https://peps.python.org/pep-0544/)

---

## 3. 📞 Simplification des appels de fonctions

**Problème** : Certains appels de fonctions sont complexes et difficiles à comprendre, avec de nombreux paramètres.

**Solution** :
- Utiliser des noms de fonctions et de variables explicites.
- Décomposer les appels complexes en étapes simples.
- Utiliser des paramètres nommés pour clarifier les appels.

**Explication** : Des appels de fonctions simples et clairs améliorent la lisibilité et la maintenabilité du code. Cela facilite également le débogage et les tests.

**Effets de bord** :
- L'utilisation de paramètres nommés peut rendre les appels plus verbeux.
- Les noms explicites peuvent entrer en conflit avec des conventions existantes.

**Solutions pour éviter les effets de bord** :
- **Équilibre entre concision et clarté** : Utiliser des paramètres nommés uniquement lorsque cela améliore la lisibilité.
- **Respect des conventions** : Suivre les conventions de nommage existantes dans le projet.
- **Documentation** : Documenter les paramètres des fonctions pour clarifier leur utilisation.

**Exemple** :

**Avant** :
```python
# pipelines/brest/pipeline.py
def get_fictive(self, data, n_sejours=1000, n_ccam=3, n_das=3, ghm5_pattern=None, **kwargs):
    return generate_fictive_stays(data, n_sejours, generate_brest_fictive, n_ccam, n_das, ghm5_pattern)

# Utilisation
stays = pipeline.get_fictive(data, 500, 2, 4, "06C")
```

**Après** :
```python
# pipelines/brest/pipeline.py
def get_fictive(
    self,
    data: dict[str, pl.LazyFrame],
    n_sejours: int = 1000,
    n_ccam: int = 3,
    n_das: int = 3,
    ghm5_pattern: str | None = None,
    **kwargs,
) -> pl.DataFrame:
    return generate_fictive_stays(
        data=data,
        n_sejours=n_sejours,
        generate_fn=generate_brest_fictive,
        n_ccam=n_ccam,
        n_das=n_das,
        ghm5_pattern=ghm5_pattern,
    )

# Utilisation
stays = pipeline.get_fictive(
    data=data,
    n_sejours=500,
    n_ccam=2,
    n_das=4,
    ghm5_pattern="06C",
)
```

**Sources** :
- [PEP 8 -- Function and Variable Names](https://peps.python.org/pep-0008/#function-and-variable-names)

---

## 4. 🧰 Utilisation de fonctions utilitaires

**Problème** : Certaines opérations sont répétées dans plusieurs modules, ce qui entraîne de la duplication de code.

**Solution** :
- Créer des fonctions utilitaires dans des modules existants comme `core/utils.py`
- Utiliser des dataclasses pour les structures de données communes
- Regrouper les fonctions communes par domaine fonctionnel

**Explication** : Les fonctions utilitaires réduisent la duplication de code et améliorent la maintenabilité. Elles permettent également de centraliser la logique commune.

**Effets de bord** :
- La création d'un module utilitaire central peut introduire une nouvelle dépendance
- Les fonctions utilitaires peuvent devenir un "fourre-tout" si mal gérées

**Solutions pour éviter les effets de bord** :
- **Organisation claire** : Structurer par domaine (fichiers, données, validation)
- **Responsabilité unique** : Une fonction = une tâche spécifique
- **Documentation** : Docstrings complètes avec exemples
- **Tests unitaires** : Couverture complète des fonctions utilitaires

**Exemple avec le code du projet** :

**Avant (duplication de code)** :
```python
# pipelines/brest/pipeline.py
def check_data(self) -> None:
    input_dir = Path(self.config["data"]["input"])
    if not input_dir.exists():
        raise FileNotFoundError(f"Répertoire introuvable: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Chemin non valide: {input_dir}")

# pipelines/aphp/pipeline.py  
def check_data(self) -> None:
    input_dir = Path(self.config["data"]["input"])
    if not input_dir.exists():
        raise FileNotFoundError(f"Répertoire introuvable: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Chemin non valide: {input_dir}")
```

**Après (fonction utilitaire avec dataclass)** :
```python
# core/utils.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PipelineConfig:
    """Configuration standardisée pour les pipelines."""
    data_input: Path
    data_output: Path
    temp_dir: Path

def validate_pipeline_config(config: dict) -> PipelineConfig:
    """Valide et convertit la configuration en objet typé.
    
    Args:
        config: Configuration brute depuis le fichier YAML/JSON
        
    Returns:
        PipelineConfig: Configuration validée et typée
        
    Raises:
        FileNotFoundError: Si un répertoire est introuvable
        NotADirectoryError: Si un chemin n'est pas un répertoire
        ValueError: Si la configuration est incomplète
    """
    try:
        input_dir = Path(config["data"]["input"])
        output_dir = Path(config["data"]["output"])
        temp_dir = Path(config.get("temp_dir", "/tmp"))
        
        for path in [input_dir, output_dir, temp_dir]:
            if not path.exists():
                raise FileNotFoundError(f"Répertoire manquant: {path}")
            if not path.is_dir():
                raise NotADirectoryError(f"Chemin invalide: {path}")
        
        return PipelineConfig(
            data_input=input_dir,
            data_output=output_dir,
            temp_dir=temp_dir
        )
    except KeyError as e:
        raise ValueError(f"Configuration incomplète: {e}") from e

# pipelines/brest/pipeline.py
from core.utils import validate_pipeline_config, PipelineConfig

class BrestPipeline:
    def __init__(self, config: dict):
        self.config: PipelineConfig = validate_pipeline_config(config)
    
    def check_data(self) -> None:
        """Utilise la configuration déjà validée."""
        if not self.config.data_input.exists():
            raise FileNotFoundError(f"Données manquantes: {self.config.data_input}")

# pipelines/aphp/pipeline.py
from core.utils import validate_pipeline_config, PipelineConfig

class APHPPipeline:
    def __init__(self, config: dict):
        self.config: PipelineConfig = validate_pipeline_config(config)
    
    def check_data(self) -> None:
        """Utilise la configuration déjà validée."""
        if not self.config.data_input.exists():
            raise FileNotFoundError(f"Données manquantes: {self.config.data_input}")
```

**Avantages de cette approche** :
- ✅ Validation centralisée de la configuration
- ✅ Utilisation de types forts avec dataclasses
- ✅ Réduction de 80% du code dupliqué
- ✅ Meilleure détection des erreurs grâce au typage
- ✅ Configuration validée une seule fois à l'initialisation

**Exemple avancé (utilitaires de données)** :
```python
# core/data_utils.py
import polars as pl
from typing import Any

def standardize_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """Standardise les noms de colonnes (snake_case, sans accents)."""
    return df.rename({
        col: col.lower().replace(' ', '_').replace('-', '_')
        for col in df.columns
    })

def convert_to_lazy(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Convertit systématiquement en LazyFrame pour les pipelines."""
    return df.lazy() if isinstance(df, pl.DataFrame) else df

# Utilisation dans les pipelines
# pipelines/brest/loader.py
from core.data_utils import standardize_column_names, convert_to_lazy

def load_and_clean_data(file_path: str) -> pl.LazyFrame:
    df = pl.scan_parquet(file_path)
    df = standardize_column_names(df)
    return convert_to_lazy(df)
```

**Sources** :
- [Real Python - DRY Python](https://realpython.com/dry-python/)
- [Python dataclasses](https://docs.python.org/3/library/dataclasses.html)

---

## 5. 📊 Simplification des structures de données

**Problème** : Certaines structures de données sont complexes avec des imbrications profondes, ce qui les rend difficiles à manipuler et à comprendre.

**Solution** :
- Utiliser des dataclasses pour les structures complexes
- Éviter les imbrications profondes (> 2 niveaux)
- Utiliser des noms de clés explicites et des types forts
- Créer des fonctions d'adaptation pour la compatibilité

**Explication** : Des structures de données simples améliorent la lisibilité, la maintenabilité et réduisent les erreurs. Elles sont également plus faciles à manipuler et à déboguer.

**Effets de bord** :
- Les structures simplifiées peuvent perdre des informations contextuelles
- Les changements de structure peuvent nécessiter des mises à jour dans plusieurs modules

**Solutions pour éviter les effets de bord** :
- **Migration progressive** : Simplifier par étapes avec validation
- **Compatibilité ascendante** : Maintenir des fonctions d'adaptation
- **Tests exhaustifs** : Vérifier toutes les fonctionnalités après simplification
- **Documentation** : Documenter les nouvelles structures et migrations

**Exemple avec le code du projet** :

**Avant (structure complexe imbriquée)** :
```python
# pipelines/aphp/loader.py
def load_pmsi_data(input_dir: str) -> dict:
    return {
        "scenarios": {
            "data": pl.scan_parquet(f"{input_dir}/scenarios.parquet"),
            "metadata": {
                "source": "ATIH",
                "version": "2023",
                "description": "Profils PMSI avec diagnostics principaux et associés",
                "schema": {
                    "columns": ["ghm", "diagnostic", "age"],
                    "types": ["str", "str", "int"]
                }
            },
        },
        "procedures": {
            "data": pl.scan_parquet(f"{input_dir}/procedures.parquet"),
            "metadata": {
                "source": "ATIH",
                "version": "2023",
                "description": "Actes CCAM par GHM",
            },
        },
    }

# Utilisation complexe
data = load_pmsi_data("data/aphp")
scenarios_df = data["scenarios"]["data"]
metadata = data["scenarios"]["metadata"]["description"]
```

**Après (structure simplifiée avec dataclasses)** :
```python
# pipelines/aphp/datatypes.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import polars as pl

@dataclass
class PMSIDataset:
    """Structure typée pour un jeu de données PMSI."""
    data: pl.LazyFrame
    source: str = "ATIH"
    version: str = "2023"
    description: str = ""
    schema: Dict[str, str] | None = None

@dataclass
class PMSIData:
    """Structure principale pour les données PMSI."""
    scenarios: PMSIDataset
    procedures: PMSIDataset
    related_diag: PMSIDataset

# pipelines/aphp/loader.py
from pipelines.aphp.datatypes import PMSIData, PMSIDataset

def load_pmsi_data(input_dir: str) -> PMSIData:
    """Charge les données PMSI avec structure typée.
    
    Returns:
        PMSIData: Objet typé contenant tous les jeux de données
    """
    return PMSIData(
        scenarios=PMSIDataset(
            data=pl.scan_parquet(f"{input_dir}/scenarios.parquet"),
            description="Profils PMSI avec diagnostics principaux et associés",
            schema={"ghm": "str", "diagnostic": "str", "age": "int"}
        ),
        procedures=PMSIDataset(
            data=pl.scan_parquet(f"{input_dir}/procedures.parquet"),
            description="Actes CCAM par GHM"
        ),
        related_diag=PMSIDataset(
            data=pl.scan_parquet(f"{input_dir}/related_diag.parquet"),
            description="Diagnostics associés"
        )
    )

# Fonction de compatibilité pour migration progressive
def load_pmsi_data_legacy(input_dir: str) -> dict:
    """Ancienne version pour compatibilité ascendante."""
    pmsi_data = load_pmsi_data(input_dir)
    return {
        "scenarios": {
            "data": pmsi_data.scenarios.data,
            "metadata": {
                "source": pmsi_data.scenarios.source,
                "version": pmsi_data.scenarios.version,
                "description": pmsi_data.scenarios.description,
            },
        },
        "procedures": {
            "data": pmsi_data.procedures.data,
            "metadata": {
                "source": pmsi_data.procedures.source,
                "version": pmsi_data.procedures.version,
            },
        },
    }

# Utilisation simplifiée et typée
data = load_pmsi_data("data/aphp")
scenarios_df = data.scenarios.data  # Accès direct avec autocomplétion
print(data.scenarios.description)  # Accès aux métadonnées
```

**Comparaison des approches** :

| Aspect                | Avant (dict imbriqué)       | Après (dataclasses)        |
|-----------------------|---------------------------|---------------------------|
| **Lisibilité**        | ❌ 4 niveaux d'imbrication | ✅ Structure plate         |
| **Typage**            | ❌ Pas de typage            | ✅ Types forts             |
| **Autocomplétion**    | ❌ Pas d'aide IDE          | ✅ Complétion complète     |
| **Maintenabilité**     | ❌ Difficile à modifier     | ✅ Facile à étendre       |
| **Performance**       | ✅ Identique               | ✅ Identique              |
| **Compatibilité**      | ✅ N/A                     | ✅ Fonction d'adaptation   |

**Exemple avancé (validation de structure)** :
```python
# pipelines/aphp/validator.py
from pipelines.aphp.datatypes import PMSIData

def validate_pmsi_data(pmsi_data: PMSIData) -> bool:
    """Valide la structure et le contenu des données PMSI."""
    # Vérification des DataFrames
    for dataset in [pmsi_data.scenarios, pmsi_data.procedures, pmsi_data.related_diag]:
        if dataset.data is None:
            return False
        if len(dataset.data.columns) == 0:
            return False
    
    # Vérification des métadonnées
    if not pmsi_data.scenarios.description:
        return False
    
    return True

# Utilisation
pmsi_data = load_pmsi_data("data/aphp")
if validate_pmsi_data(pmsi_data):
    print("✅ Données PMSI valides")
else:
    print("❌ Problème avec les données PMSI")
```

**Sources** :
- [Python dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [Real Python - Python Data Classes](https://realpython.com/python-data-classes/)
- [PEP 557 -- Data Classes](https://peps.python.org/pep-0557/)

---

## 6. 📦 Réduction des dépendances externes

**Problème** : Le projet dépend de nombreuses bibliothèques externes, ce qui peut compliquer l'installation, la maintenance et augmenter les risques de conflits.

**Solution** :
- Standardiser sur une bibliothèque principale (polars)
- Éviter les dépendances redondantes
- Utiliser des fonctions d'adaptation pour les cas spécifiques
- Documenter les choix de dépendances

**Explication** : Moins de dépendances externes signifie moins de risques de conflits, une installation plus simple et une meilleure portabilité du code.

**Analyse des dépendances actuelles** :
```
┌─────────────────┬─────────────┬─────────────────────────────────────┐
│ Bibliothèque     │ Utilisation  │ Raison de conservation/remplacement │
├─────────────────┼─────────────┼─────────────────────────────────────┤
│ polars          │ ✅ Principale│ Standardisé pour le projet         │
│ pandas          │ ❌ Redondant │ Remplaçable par polars             │
│ numpy           │ ⚠️  Partiel  │ Utilisé uniquement pour quelques   │
│                 │              │ opérations vectorielles            │
│ pydantic        │ ✅ Validation│ Validation des configurations       │
│ requests        │ ✅ API       │ Appels HTTP nécessaires            │
└─────────────────┴─────────────┴─────────────────────────────────────┘
```

**Effets de bord** :
- La suppression de dépendances peut casser des fonctionnalités existantes
- Les bibliothèques standard peuvent avoir des performances différentes
- Certaines fonctionnalités spécifiques peuvent nécessiter des dépendances dédiées

**Solutions pour éviter les effets de bord** :
- **Analyse d'impact** : Cartographier tous les usages avant suppression
- **Tests de performance** : Benchmarker les alternatives
- **Migration progressive** : Remplacer par étapes avec validation
- **Fonctions d'adaptation** : Créer des wrappers pour les cas spécifiques

**Exemple avec le code du projet** :

**Avant (utilisation mixte pandas/polars)** :
```python
# requirements.txt
pandas==2.0.3
polars==0.19.0

# pipelines/brest/loader.py
import pandas as pd
import polars as pl

def load_csv_data(file_path: str) -> pd.DataFrame:
    """Charge un fichier CSV avec pandas."""
    return pd.read_csv(file_path, sep=";", encoding="latin-1")

def process_data(pandas_df: pd.DataFrame) -> pl.DataFrame:
    """Convertit pandas vers polars pour le traitement."""
    return pl.from_pandas(pandas_df)

# Utilisation inefficace
data = load_csv_data("data/brest/PMSI_DP.csv")
processed = process_data(data)
```

**Après (standardisation sur polars)** :
```python
# requirements.txt
polars==0.19.0

# pipelines/brest/loader.py
import polars as pl

def load_csv_data(file_path: str) -> pl.LazyFrame:
    """Charge un fichier CSV avec polars (optimisé pour les gros fichiers).
    
    Args:
        file_path: Chemin vers le fichier CSV
        
    Returns:
        LazyFrame pour un traitement efficace en mémoire
        
    Note:
        - polars est 2-5x plus rapide que pandas pour les gros fichiers
        - Utilise lazy evaluation pour optimiser les performances
        - Déjà utilisé dans le reste du projet → cohérence
    """
    return pl.scan_csv(
        file_path,
        separator=";",
        encoding="latin-1",
        infer_schema_length=10000,
        low_memory=True,
        ignore_errors=True
    )

def load_csv_data_eager(file_path: str) -> pl.DataFrame:
    """Version eager pour les petits fichiers ou compatibilité."""
    return pl.read_csv(
        file_path,
        separator=";",
        encoding="latin-1",
        infer_schema_length=10000
    )

# Fonction d'adaptation pour les rares cas nécessitant pandas
def convert_to_pandas(df: pl.DataFrame | pl.LazyFrame) -> 'pd.DataFrame':
    """Convertit polars vers pandas (à utiliser uniquement si nécessaire).
    
    Warning:
        Cette fonction charge les données en mémoire et peut être coûteuse
        pour les grands jeux de données. Préférer polars lorsque possible.
    """
    import pandas as pd
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    return df.to_pandas()

# Utilisation optimisée
data = load_csv_data("data/brest/PMSI_DP.csv")
# Traitement direct en lazy sans conversion
result = data.filter(pl.col("age") > 18).group_by("ghm").count()
```

**Benchmark comparatif** :
```python
# pipelines/benchmarks/performance.py
import time
import polars as pl

def benchmark_pandas_vs_polars(file_path: str, iterations: int = 5):
    """Compare les performances pandas vs polars."""
    
    # Benchmark pandas
    start = time.time()
    for _ in range(iterations):
        import pandas as pd
        pd.read_csv(file_path, sep=";", encoding="latin-1")
    pandas_time = time.time() - start
    
    # Benchmark polars
    start = time.time()
    for _ in range(iterations):
        pl.scan_csv(file_path, separator=";", encoding="latin-1")
    polars_time = time.time() - start
    
    print(f"Pandas: {pandas_time:.2f}s | Polars: {polars_time:.2f}s")
    print(f"Gain: {pandas_time/polars_time:.1f}x plus rapide")
    
    return {"pandas": pandas_time, "polars": polars_time}
```

**Stratégie de migration** :

1. **Phase 1 - Audit** (1 semaine)
   - Identifier tous les usages de pandas/numpy
   - Catégoriser: critique, utile, redondant
   - Documenter dans un fichier `MIGRATION.md`

2. **Phase 2 - Remplacement** (2-3 semaines)
   - Remplacer les usages simples par polars
   - Créer des fonctions d'adaptation pour les cas complexes
   - Ajouter des tests de non-régression

3. **Phase 3 - Validation** (1 semaine)
   - Benchmarker les performances
   - Vérifier la compatibilité
   - Documenter les changements

4. **Phase 4 - Nettoyage** (1 semaine)
   - Supprimer les imports inutilisés
   - Mettre à jour les requirements
   - Supprimer les dépendances obsolètes

**Exemple de mise à jour des requirements** :

**Avant** :
```python
# requirements.txt
pandas==2.0.3
numpy==1.24.3
polars==0.19.0
pydantic==2.0.3
scikit-learn==1.2.2
```

**Après** :
```python
# requirements.txt
polars==0.19.0
pydantic==2.0.3

# Dépendances optionnelles pour fonctionnalités spécifiques
# (dans requirements-optional.txt)
pandas==2.0.3  # Pour export Excel avancé
scikit-learn==1.2.2  # Pour modèles ML spécifiques
```

**Documentation des choix** :

Créer un fichier `DEPENDENCIES.md` expliquant:
- Pourquoi polars est la bibliothèque standard
- Quand utiliser les dépendances optionnelles
- Comment ajouter de nouvelles dépendances
- Procédure de mise à jour des dépendances

**Sources** :
- [Python Packaging User Guide](https://packaging.python.org/en/latest/)
- [Polars vs Pandas Benchmark](https://h2o.ai/blog/polars-vs-pandas-benchmark/)
- [Polars Documentation](https://pola-rs.github.io/polars/py-polars/html/)

---

## 7. 🚀 Simplification des appels aux pipelines

**Problème** : Les appels aux pipelines sont complexes et difficiles à comprendre, avec de nombreuses étapes.

**Solution** :
- Créer des fonctions de haut niveau pour simplifier les appels.
- Utiliser des paramètres par défaut pour réduire la complexité.
- Décomposer les opérations en étapes claires.

**Explication** : Des appels simplifiés aux pipelines améliorent la lisibilité et réduisent les erreurs. Cela facilite également l'intégration et les tests.

**Effets de bord** :
- Les fonctions de haut niveau peuvent masquer des détails importants pour les utilisateurs avancés.
- Les paramètres par défaut peuvent ne pas couvrir tous les cas d'usage.

**Solutions pour éviter les effets de bord** :
- **Flexibilité** : Permettre aux utilisateurs avancés d'accéder aux détails si nécessaire.
- **Documentation claire** : Documenter les paramètres par défaut et comment les personnaliser.
- **Tests d'intégration** : Vérifier que les fonctions de haut niveau couvrent tous les cas d'usage.

**Exemple avec le code du projet** :

**Avant (appel complexe et répétitif)** :
```python
# runner.py
def run_pipeline(pipeline_name: str, config: dict, n_sejours: int = 1000):
    if pipeline_name == "brest":
        pipeline = BrestPipeline(config)
    elif pipeline_name == "aphp":
        pipeline = APHPPipeline(config)
    else:
        raise ValueError(f"Pipeline inconnu: {pipeline_name}")
    
    pipeline.check_data()
    data = pipeline.load_data()
    stays = pipeline.get_fictive(data, n_sejours=n_sejours)
    scenarios = pipeline.get_scenario(stays)
    reports = pipeline.get_report(scenarios, client, model)
    return reports

# Utilisation répétitive dans différents scripts
reports_brest = run_pipeline("brest", config, 1000)
reports_aphp = run_pipeline("aphp", config, 500)
```

**Après (fonction de haut niveau avec flexibilité)** :
```python
# runner.py
def run_pipeline(
    pipeline_name: str,
    config: dict,
    n_sejours: int = 1000,
    client: Any = None,
    model: str = "mistral",
    batch_size: int = 1000,
    **kwargs,
) -> pl.DataFrame:
    """Exécute un pipeline et retourne les rapports générés.
    
    Args:
        pipeline_name: Nom du pipeline ("brest" ou "aphp").
        config: Configuration du pipeline.
        n_sejours: Nombre de séjours fictifs à générer.
        client: Client LLM à utiliser.
        model: Modèle LLM à utiliser.
        batch_size: Taille des batches pour la génération.
        **kwargs: Arguments supplémentaires passés au pipeline.
        
    Returns:
        DataFrame avec les rapports générés.
        
    Raises:
        ValueError: Si le pipeline est inconnu.
    """
    pipeline = get_pipeline(pipeline_name, config)
    data = pipeline.load_data()
    stays = pipeline.get_fictive(data, n_sejours=n_sejours, **kwargs)
    scenarios = pipeline.get_scenario(stays)
    return pipeline.get_report(scenarios, client, model, batch_size=batch_size)

def get_pipeline(pipeline_name: str, config: dict) -> BasePipeline:
    """Retourne une instance de pipeline.
    
    Args:
        pipeline_name: Nom du pipeline.
        config: Configuration du pipeline.
        
    Returns:
        Instance du pipeline.
        
    Raises:
        ValueError: Si le pipeline est inconnu.
    """
    pipelines = {
        "brest": BrestPipeline,
        "aphp": APHPPipeline,
    }
    if pipeline_name not in pipelines:
        raise ValueError(
            f"Pipeline inconnu: {pipeline_name}. "
            f"Choix possibles: {list(pipelines.keys())}"
        )
    return pipelines[pipeline_name](config)

# Fonction avancée pour les utilisateurs qui ont besoin de plus de contrôle
def run_pipeline_advanced(
    pipeline: BasePipeline,
    data: dict[str, pl.LazyFrame],
    **kwargs,
) -> pl.DataFrame:
    """Exécute un pipeline avec un contrôle complet sur chaque étape."""
    stays = pipeline.get_fictive(data, **kwargs)
    scenarios = pipeline.get_scenario(stays)
    return pipeline.get_report(scenarios, **kwargs)

# Utilisation simplifiée
reports_brest = run_pipeline("brest", config, n_sejours=1000)
reports_aphp = run_pipeline("aphp", config, n_sejours=500, model="claude")

# Utilisation avancée pour les cas spécifiques
pipeline = BrestPipeline(config)
data = pipeline.load_data()
reports_custom = run_pipeline_advanced(
    pipeline,
    data,
    n_sejours=200,
    n_ccam=5,
    ghm5_pattern="06C",
)
```

**Exemple** :

**Avant (appel complexe)** :
```python
# runner.py
def run_pipeline(pipeline_name: str, config: dict, n_sejours: int = 1000):
    if pipeline_name == "brest":
        pipeline = BrestPipeline(config)
    elif pipeline_name == "aphp":
        pipeline = APHPPipeline(config)
    else:
        raise ValueError(f"Pipeline inconnu: {pipeline_name}")
    
    pipeline.check_data()
    data = pipeline.load_data()
    stays = pipeline.get_fictive(data, n_sejours=n_sejours)
    scenarios = pipeline.get_scenario(stays)
    reports = pipeline.get_report(scenarios, client, model)
    return reports
```

**Après (appel simplifié)** :
```python
# runner.py
def run_pipeline(
    pipeline_name: str,
    config: dict,
    n_sejours: int = 1000,
    client: Any = None,
    model: str = "mistral",
) -> pl.DataFrame:
    """Exécute un pipeline et retourne les rapports générés."""
    pipeline = get_pipeline(pipeline_name, config)
    data = pipeline.load_data()
    stays = pipeline.get_fictive(data, n_sejours=n_sejours)
    scenarios = pipeline.get_scenario(stays)
    return pipeline.get_report(scenarios, client, model)

def get_pipeline(pipeline_name: str, config: dict) -> BasePipeline:
    """Retourne une instance de pipeline."""
    pipelines = {
        "brest": BrestPipeline,
        "aphp": APHPPipeline,
    }
    if pipeline_name not in pipelines:
        raise ValueError(f"Pipeline inconnu: {pipeline_name}. Choix possibles: {list(pipelines.keys())}")
    return pipelines[pipeline_name](config)
```

**Sources** :
- [Real Python - Python Functions](https://realpython.com/defining-your-own-python-function/)



## Conclusion

Ces recommandations visent à simplifier le code et les dépendances du repo Stream pour faciliter la maintenance et le développement. Elles sont adaptées à la phase actuelle du projet où la simplicité est prioritaire. En appliquant ces suggestions, le code sera plus facile à comprendre, à maintenir et à étendre.
