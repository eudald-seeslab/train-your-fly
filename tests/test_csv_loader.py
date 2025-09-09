import os
import pandas as pd
import time

from connectome.core.csv_loader import CSVLoader


def test_csv_loader_cache(tmp_path):
    csv_path = os.path.join(tmp_path, "sample.csv")
    df_in = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df_in.to_csv(csv_path, index=False)

    loader = CSVLoader()

    # First read – creates pickle
    df1 = loader.read_csv(csv_path)
    assert df1.equals(df_in)

    pkl_path = csv_path.replace(".csv", ".pkl")
    assert os.path.exists(pkl_path)

    # Touch csv to be older than pickle
    os.utime(pkl_path, None)  # update mtime of pickle to now
    time.sleep(0.1)
    df_in.iloc[0, 0] = 99  # change dataframe but don't rewrite csv
    df2 = loader.read_csv(csv_path)
    # Should still return original content from pickle cache
    assert df2.equals(df1)
