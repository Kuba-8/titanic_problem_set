import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

def summarize_data(d_frame, name_of_df="Data Frame"):
    """

    This function computes a detailed summary and overview of a given pandas dataframe.

    """
    print(f"Overview of {name_of_df}")
    print(f"The first 5 rows of {name_of_df}")
    print(d_frame.head())
    print(f"{name_of_df} information")
    d_frame.info()
    print(f"Summary statistics for {name_of_df}")
    print(d_frame.describe())

summarize_data(train_df)
summarize_data(test_df)


def missing_summary(d_frame):
    """
    
    Function that provides a summary of the missing values within a given dataframe, including:
    - The total number
    - The percentage of missing values
    - The datatype of each column
    
    """
    total = d_frame.isnull().sum()
    percent = (d_frame.isnull().sum()/d_frame.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in d_frame.columns:
        dtype = str(d_frame[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    missings_d_frame = np.transpose(tt)
    return missings_d_frame

missing_summary(train_df)
missing_summary(test_df)



def degree_of_uniformity(d_frame):
    """
    
    Function for the percentage contribution and value count of the most frequent value within a particular column
    
    """
    total = d_frame.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in d_frame.columns:
        try:
            itm = d_frame[col].value_counts().index[0]
            val = d_frame[col].value_counts().values[0]
            items.append(itm)
            vals.append(val)
        except Exception as ex:
            print(ex)
            items.append(0)
            vals.append(0)
        continue
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    uniformity_calc = np.transpose(tt)
    return uniformity_calc

degree_of_uniformity(train_df)
degree_of_uniformity(test_df)




def unique_value_count(d_frame):
    """
    
    Function that calculates the total number of unique values within a dataframe
    
    """
    total = d_frame.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in d_frame.columns:
        unique = d_frame[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    uniques_total = np.transpose(tt)
    return uniques_total

unique_value_count(train_df)
unique_value_count(test_df)
