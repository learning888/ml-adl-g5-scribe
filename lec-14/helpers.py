#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def preprocess_data(df):
    """TODO: Docstring for preprocess_data.

    :df: TODO
    :returns: TODO

    """
    tmp_df = df.copy()
    get_categoricals = tmp_df.dtypes[
        np.where(tmp_df.dtypes.values == 'object', True, False)
    ]

    for i in get_categoricals.index:
        tmp_dummies = pd.get_dummies(
            tmp_df[i], drop_first=True, prefix=i
        )
        tmp_df = tmp_df.drop(columns=i)
        tmp_df = pd.concat(
            [tmp_df, tmp_dummies], axis=1
        )

    return tmp_df
