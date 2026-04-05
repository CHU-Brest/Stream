import argparse
import sys

from runner import run


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="stream",
        description="Génération de comptes rendus d'hospitalisation synthétiques.",
    )
    parser.add_argument(
        "pipeline",
        choices=["brest", "aphp"],
        help="Pipeline source de données à utiliser.",
    )
    parser.add_argument(
        "--client",
        choices=["ollama", "claude", "mistral"],
        default="ollama",
        help="Type de client LLM (défaut : ollama).",
    )
    parser.add_argument(
        "--n-sejours",
        type=int,
        default=1000,
        metavar="N",
        help="Nombre de séjours fictifs à générer (défaut : 1000).",
    )
    parser.add_argument(
        "--n-ccam",
        type=int,
        default=1,
        metavar="N",
        help="Nombre max d'actes CCAM par séjour (défaut : 1).",
    )
    parser.add_argument(
        "--n-das",
        type=int,
        default=5,
        metavar="N",
        help="Nombre max de diagnostics associés par séjour (défaut : 5).",
    )
    parser.add_argument(
        "--ghm5",
        default=None,
        metavar="PATTERN",
        help="Filtre optionnel sur les codes GHM5 (ex. '06C').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="Nombre de CRH avant flush Parquet (défaut : 1000).",
    )

    args = parser.parse_args()

    try:
        run(
            pipeline_name=args.pipeline,
            client_type=args.client,
            n_sejours=args.n_sejours,
            n_ccam=args.n_ccam,
            n_das=args.n_das,
            ghm5_pattern=args.ghm5,
            batch_size=args.batch_size,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Erreur : {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
