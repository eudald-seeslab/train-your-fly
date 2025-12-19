import os
import pandas as pd


class CSVLoader:
    """Lightweight helper that wraps ``pandas.read_csv`` with an on-disk pickle
    cache for much faster repeat load times.

    Behaviour is *identical* to the helper that used to live inside
    ``DataProcessor`` so that existing code sees no functional change.
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache

    def read_csv(self, csv_path: str, **read_csv_kwargs):
        """Read *csv_path* honouring the pickle cache when enabled."""
        if not self.use_cache:
            return pd.read_csv(csv_path, **read_csv_kwargs)

        pkl_path = csv_path.replace(".csv", ".pkl")
        try:
            if os.path.exists(pkl_path) and os.path.getmtime(
                pkl_path
            ) >= os.path.getmtime(csv_path):
                return pd.read_pickle(pkl_path)
        except (OSError, FileNotFoundError):
            # Fall back to reading CSV if we cannot stat the files (e.g. on some
            # network file systems).
            pass

        df = pd.read_csv(csv_path, **read_csv_kwargs)

        # Store pickle for next time; ignore any failure (e.g., read-only FS)
        try:
            df.to_pickle(pkl_path)
        except Exception:
            pass

        return df
