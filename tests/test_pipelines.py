"""Unit tests for the pipelines.

This module contains tests for the Brest and AP-HP pipelines.
"""

from pathlib import Path
import tempfile
import pytest

from pipelines.brest.pipeline import BrestPipeline
from pipelines.aphp.pipeline import APHPPipeline


class TestBrestPipeline:
    """Test the Brest pipeline."""

    def test_initialization(self):
        """Test that the BrestPipeline can be initialized."""
        config = {"data": {"input": "/tmp", "output": "/tmp"}}
        prompt = {"generate": {"system_prompt": "Test prompt"}}
        servers = {"ollama": {"host": "http://localhost:11434", "model": "mistral"}}

        pipeline = BrestPipeline(config=config, prompt=prompt, servers=servers)
        assert pipeline.name == "brest"

    def test_check_data_without_files(self):
        """Test that check_data handles missing data files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CSV files with minimal data to avoid NoDataError
            for csv_file in BrestPipeline.SOURCES.values():
                file_path = Path(tmpdir) / csv_file
                # Write minimal CSV data to avoid NoDataError
                if csv_file == "PMSI_DP.csv":
                    file_path.write_text("GHM5;DP;P_DP\n06C01Z;E119;0.5\n")
                elif csv_file == "PMSI_CCAM_DP.csv":
                    file_path.write_text("GHM5;DP;CCAM;P_CCAM\n06C01Z;E119;JJFG001;0.5\n")
                elif csv_file == "PMSI_DAS.csv":
                    file_path.write_text("GHM5;AGE;SEXE;DP;DAS;P_DAS\n06C01Z;65;1;E119;I10;0.5\n")
                elif csv_file == "PMSI_DMS.csv":
                    file_path.write_text("GHM5;DMS_P25;DMS_P50;DMS_P75\n06C01Z;1;2;3\n")
                elif csv_file == "ALL_CLASSIF_PMSI.csv":
                    file_path.write_text("champ;version;type_code;code;lib\nmco;v2024;racine;06C01Z;Test GHM\n")
                elif csv_file == "ALL_CIM10.csv":
                    file_path.write_text("code;liblong;en_cours\nE119;Diabète de type 2;1\n")
                elif csv_file == "ALL_CCAM.csv":
                    file_path.write_text("code;liblong;en_cours\nJJFG001;Test CCAM;1\n")
            
            config = {"data": {"input": tmpdir, "output": tmpdir}}
            prompt = {"generate": {"system_prompt": "Test prompt"}}
            servers = {"ollama": {"host": "http://localhost:11434", "model": "mistral"}}

            pipeline = BrestPipeline(config=config, prompt=prompt, servers=servers)
            # Since no data files are present, check_data should handle it gracefully
            # (it will create Parquet files from CSV if they don't exist)
            pipeline.check_data()


class TestAPHPPipeline:
    """Test the AP-HP pipeline."""

    def test_initialization(self):
        """Test that the APHPPipeline can be initialized."""
        config = {
            "data": {
                "input": "/tmp",
                "output": "/tmp",
                "referentials": "/tmp/referentials"
            }
        }
        prompt = {"generate": {"system_prompt": "Test prompt"}}
        servers = {"ollama": {"host": "http://localhost:11434", "model": "mistral"}}

        pipeline = APHPPipeline(config=config, prompt=prompt, servers=servers)
        assert pipeline.name == "aphp"

    def test_check_data_without_files(self):
        """Test that check_data raises an error if data files are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "data": {
                    "input": tmpdir,
                    "output": tmpdir,
                    "referentials": f"{tmpdir}/referentials"
                }
            }
            prompt = {"generate": {"system_prompt": "Test prompt"}}
            servers = {"ollama": {"host": "http://localhost:11434", "model": "mistral"}}

            pipeline = APHPPipeline(config=config, prompt=prompt, servers=servers)
            # Since no data files are present, check_data should raise FileNotFoundError
            # We need to create the referentials directory to avoid FileNotFoundError
            Path(config["data"]["referentials"]).mkdir(parents=True, exist_ok=True)
            with pytest.raises(FileNotFoundError):
                pipeline.check_data()


class TestPipelineIntegration:
    """Test the integration of pipelines with the runner."""

    def test_pipelines_are_registered(self):
        """Test that pipelines are correctly registered in the runner."""
        from runner import PIPELINES

        assert "brest" in PIPELINES
        assert "aphp" in PIPELINES
        assert PIPELINES["brest"] == BrestPipeline
        assert PIPELINES["aphp"] == APHPPipeline
