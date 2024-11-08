import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import RandomForestClassifier
from titanic.config import TEST_PATH, TRAIN_PATH

# Preliminary data inspection
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




# Exploratory data analysis
def all_df(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    
    """

    train_copy = train_df.copy()
    test_copy = test_df.copy()



    all_df = pd.concat([train_copy, test_copy], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df.Survived.isna(), "set"] = "test"
    all_df["Family Size"] = all_df["SibSp"] + all_df["Parch"] + 1
    
    all_df["Age Interval"] = 0.0
    all_df.loc[ all_df['Age'] <= 16, 'Age Interval']  = 0
    all_df.loc[(all_df['Age'] > 16) & (all_df['Age'] <= 32), 'Age Interval'] = 1
    all_df.loc[(all_df['Age'] > 32) & (all_df['Age'] <= 48), 'Age Interval'] = 2
    all_df.loc[(all_df['Age'] > 48) & (all_df['Age'] <= 64), 'Age Interval'] = 3
    all_df.loc[ all_df['Age'] > 64, 'Age Interval'] = 4
    all_df['Fare Interval'] = 0.0
    all_df.loc[ all_df['Fare'] <= 7.91, 'Fare Interval'] = 0
    all_df.loc[(all_df['Fare'] > 7.91) & (all_df['Fare'] <= 14.454), 'Fare Interval'] = 1
    all_df.loc[(all_df['Fare'] > 14.454) & (all_df['Fare'] <= 31), 'Fare Interval']   = 2
    all_df.loc[ all_df['Fare'] > 31, 'Fare Interval'] = 3
    all_df["Sex_Pclass"] = all_df.apply(lambda row: row['Sex'][0].upper() + "_C" + str(row["Pclass"]), axis=1)
    all_df[["Family Name", "Title", "Given Name", "Maiden Name"]] = all_df.apply(lambda row: parse_names(row), axis=1)
    all_df = assign_family_type_and_title(all_df)
    return all_df

def train_dataframe(train_df):
    train_df = train_df.copy()
    train_df["Family Size"] = train_df["SibSp"] + train_df["Parch"] + 1
    train_df["Age Interval"] = 0.0
    train_df.loc[train_df['Age'] <= 16, 'Age Interval']  = 0
    train_df.loc[(train_df['Age'] > 16) & (train_df['Age'] <= 32), 'Age Interval'] = 1
    train_df.loc[(train_df['Age'] > 32) & (train_df['Age'] <= 48), 'Age Interval'] = 2
    train_df.loc[(train_df['Age'] > 48) & (train_df['Age'] <= 64), 'Age Interval'] = 3
    train_df.loc[ train_df['Age'] > 64, 'Age Interval'] = 4
    train_df['Fare Interval'] = 0.0
    train_df.loc[ train_df['Fare'] <= 7.91, 'Fare Interval'] = 0
    train_df.loc[(train_df['Fare'] > 7.91) & (train_df['Fare'] <= 14.454), 'Fare Interval'] = 1
    train_df.loc[(train_df['Fare'] > 14.454) & (train_df['Fare'] <= 31), 'Fare Interval']   = 2
    train_df.loc[ train_df['Fare'] > 31, 'Fare Interval'] = 3
    train_df["Sex_Pclass"] = train_df.apply(lambda row: row['Sex'][0].upper() + "_C" + str(row["Pclass"]), axis=1)
    train_df[["Family Name", "Title", "Given Name", "Maiden Name"]] = train_df.apply(lambda row: parse_names(row), axis=1)
    train_df = assign_family_type_and_title(train_df)
    train_df[['Titles', 'Sex', 'Survived']].groupby(['Titles', 'Sex'], as_index=False).mean()
    return train_df


def assign_family_type_and_title(df: pd.DataFrame) -> pd.DataFrame:
    """

    """
    df["Family Type"] = df["Family Size"]
    df.loc[df["Family Size"] == 1, "Family Type"] = "Single"
    df.loc[(df["Family Size"] > 1) & (df["Family Size"] < 5), "Family Type"] = "Small"
    df.loc[df["Family Size"] >= 5, "Family Type"] = "Large"
    df["Titles"] = df['Title']
    #unify `Miss`
    df['Titles'] = df['Titles'].replace('Mlle.', 'Miss.')
    df['Titles'] = df['Titles'].replace('Ms.', 'Miss.')
    #unify `Mrs`
    df['Titles'] = df['Titles'].replace('Mme.', 'Mrs.')
    # unify Rare
    df['Titles'] = df['Titles'].replace(['Lady.', 'the Countess.','Capt.', 'Col.','Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Rare')
    return df



def calculate_survival_rate_by_title_sex(df: pd.DataFrame) -> pd.DataFrame:
    """

    """
    return df[['Titles', 'Sex', 'Survived']].groupby(['Titles', 'Sex'], as_index=False).mean()


def parse_names(row):
    try:
        text = row["Name"]
        split_text = text.split(",")
        family_name = split_text[0]
        next_text = split_text[1]
        split_text = next_text.split(".")
        title = (split_text[0] + ".").lstrip().rstrip()
        next_text = split_text[1]
        if "(" in next_text:
            split_text = next_text.split("(")
            given_name = split_text[0]
            maiden_name = split_text[1].rstrip(")")
            return pd.Series([family_name, title, given_name, maiden_name])
        else:
            given_name = next_text
            return pd.Series([family_name, title, given_name, None])
    except Exception as ex:
        print(f"Exception: {ex}")

# baseline
def map_sex_column(datasets):
    """
    Maps the 'Sex' column in the provided datasets to numeric values.
    'female' is mapped to 1 and 'male' is mapped to 0.
    
    Parameters:
    datasets (list of pd.DataFrame): List of dataframes to modify

    Returns:
    None: The function modifies the dataframes in place.
    """
    for dataset in datasets:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

def prepare_training_data(train, valid, predictors, target):
    """
    Prepares the training and validation data for model input.
    
    Parameters:
    train (pd.DataFrame): Training dataset
    valid (pd.DataFrame): Validation dataset
    predictors (list of str): List of feature column names
    target (str): Name of the target column
    
    Returns:
    tuple: A tuple containing train_X, train_Y, valid_X, valid_Y
    - train_X (pd.DataFrame): Training features
    - train_Y (np.array): Training target values
    - valid_X (pd.DataFrame): Validation features
    - valid_Y (np.array): Validation target values
    """
    train_X = train[predictors]
    train_Y = train[target].values
    valid_X = valid[predictors]
    valid_Y = valid[target].values
    return train_X, train_Y, valid_X, valid_Y



def train_and_predict(train_X, train_Y, valid_X, 
                      n_estimators=100, criterion="gini", random_state=42):
    """
    Trains a RandomForestClassifier on the training data and returns predictions 
    for both the training and validation datasets.
    
    Parameters:
    train_X (pd.DataFrame): Training features
    train_Y (np.array): Training target values
    valid_X (pd.DataFrame): Validation features
    n_estimators (int): Number of trees in the forest (default=100)
    criterion (str): The function to measure the quality of a split (default="gini")
    random_state (int): Controls the randomness of the estimator (default=42)
    
    Returns:
    tuple: A tuple containing predictions for the training and validation data
    - preds_tr (np.array): Predictions for the training dataset
    - preds (np.array): Predictions for the validation dataset
    """
    # Initialize the RandomForestClassifier
    clf = RandomForestClassifier(n_jobs=-1, 
                                 random_state=random_state,
                                 criterion=criterion,
                                 n_estimators=n_estimators,
                                 verbose=False)
    
    # Train the model
    clf.fit(train_X, train_Y)
    
    # Generate predictions for the training and validation sets
    preds_tr = clf.predict(train_X)
    preds = clf.predict(valid_X)
    
    return preds_tr, preds

#
def plot_count_pairs(data_df, feature, title, hue="set"):
    color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.countplot(x=feature, data=data_df, hue=hue, palette= color_list)
    plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    ax.set_title(f"Number of passengers / {title}")
    filename = f"exports/{feature}_{title}_count.png"
    plt.savefig(filename, dpi=300)
    plt.show()    