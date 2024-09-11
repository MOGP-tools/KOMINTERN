
import os
import pandas as pd
import numpy as np


def compare_df(df_ref, df, tolerance = 1e-8):
    """
    compute delta between two pandas.df
    1) check label
    2) check value
    """

    is_ok = True

    df_ref_col = list(df_ref.columns)
    df_col = list(df.columns)
    if df_ref_col != df_col :
        deltaP1 = list(set(df_ref_col) - set(list(df_col)))
        deltaP2 = list(set(df_col) - set(list(df_ref_col)))
        print("Labels vary between ref / test. Label trooble :\n", deltaP1, "\n", deltaP2)

        intersection = list(set(df_ref_col) & set(df_col))
        df_ref = df_ref[intersection]
        df     = df[intersection]
        
        is_ok = False

    if np.allclose(df_ref.values, df.values, atol=tolerance):
        if is_ok :
            print("\n *** The pandas dataFrames are identical\n")
            return True, ""
        else :
            msg = " *** The new pandas file adds some new labels, but the values for the old labels remain identical.\n"
            print("\n", msg)
            return False, msg

    else :
        diff = np.abs(df_ref.values - df.values)
        non_equal_indices = np.where(diff > tolerance)

        diff_df = pd.DataFrame(index=df_ref.index, columns=df_ref.columns)
        diff_df.values[non_equal_indices] = df_ref.values[non_equal_indices] - df.values[non_equal_indices]
        mask = diff_df.notnull()

        def dropna(df, toPrint):
            df_p = df.dropna(axis=0, how='all')
            df_p = df_p.dropna(axis=1, how='all')
            print(toPrint, df_p)
        
        dropna(diff_df, "diff ref/df: \n")
        dropna(df_ref[mask], "\nvalue ref: \n")
        dropna(df[mask], "\nvalue df: \n")
        msg = "*** The pandas dataFrames are NOT identical\n"
        print("\n", msg)

        return False, msg



if __name__ == "__main__" :
    to_do()