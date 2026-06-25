import pandas as pd

def min_max_normalization(dataframe: pd.DataFrame):
    return (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())

def to_percent(number: float) -> str:
    return f"{number * 100:.2f}%"