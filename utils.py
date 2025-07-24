# loading dataset

import pandas as pd

def load_data(path):
    try:
        df = pd.read_csv(path)
        df = df[['title', 'label']].dropna()
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=['title', 'label'])
    except Exception as e:
        return pd.DataFrame(columns=['title', 'label'])