from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_environment_scripts_exist() -> None:
    assert (PROJECT_ROOT / "scripts" / "bootstrap_env.ps1").is_file()
    assert (PROJECT_ROOT / "scripts" / "run_tests.ps1").is_file()
