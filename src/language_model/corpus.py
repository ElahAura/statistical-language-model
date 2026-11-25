from pathlib import Path
import pyarrow.parquet as pq


def load_corpus(file_path: str | Path) -> list[str]:
    """
    Load a WikiText-2 Parquet file and extract its text.

    Returns a list containing one text entry per dataset row.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    table = pq.read_table(file_path)

    if "text" not in table.column_names:
        raise ValueError("Dataset does not contain a 'text' column.")

    return table.column("text").to_pylist()