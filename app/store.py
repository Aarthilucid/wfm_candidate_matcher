from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class DataStore:
    candidates: pd.DataFrame
    jobs: pd.DataFrame

_STORE: DataStore | None = None

def set_store(store: DataStore) -> None:
    global _STORE
    _STORE = store

def get_store() -> DataStore:
    if _STORE is None:
        raise RuntimeError("DataStore not loaded. Run ingestion first or call /ingest.")
    return _STORE
