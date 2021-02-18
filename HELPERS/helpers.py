# -*- coding: utf-8 -*-
"""
@author: narora62
"""
import numpy as np
import pandas as pd


def read_file(pth):
    df = pd.read_csv(pth)
    df.dropna(inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


def get_cols(_df):
    return _df.columns


if __name__ == '__main__':
    path = "~/Projects/omscs/ML/train.csv"
    df = read_file(path)
    print(df)