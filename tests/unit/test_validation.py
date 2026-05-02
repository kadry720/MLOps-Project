from src.data.validation import missing_data_directories, required_data_directories


def test_required_data_directories_returns_expected_paths(tmp_path):
    expected = [
        tmp_path / "raw",
        tmp_path / "processed",
        tmp_path / "splits",
    ]

    assert required_data_directories(tmp_path) == expected


def test_missing_data_directories_reports_only_absent_directories(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "splits").mkdir()

    assert missing_data_directories(tmp_path) == [tmp_path / "processed"]
