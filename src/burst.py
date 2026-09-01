import pandas as pd

def burst_detect(timestamps, window="10min", z_thresh=2.5):
    if timestamps.isnull().all():
        return False, None

    ts = pd.to_datetime(timestamps, errors="coerce")
    counts = ts.dt.floor(window).value_counts().sort_index()

    if counts.std() == 0:
        return False, counts

    z = (counts - counts.mean()) / counts.std()
    return (z > z_thresh).any(), counts
