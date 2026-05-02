from pathlib import Path


def required_data_directories(base_dir: str | Path = "data") -> list[Path]:
    base_path = Path(base_dir)
    return [
        base_path / "raw",
        base_path / "processed",
        base_path / "splits",
    ]


def missing_data_directories(base_dir: str | Path = "data") -> list[Path]:
    return [path for path in required_data_directories(base_dir) if not path.is_dir()]
