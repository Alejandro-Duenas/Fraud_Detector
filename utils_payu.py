"""
This module stores utility functions for the Pay Fraud & Data Scientist 
technical test.
"""

#------------------------------ 1. Libraries------------------------------------
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
import seaborn as sns
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
mlp.style.use('seaborn')
sns.set(font_scale=1.2)

#----------------------------- 3. Gobal Variables -----------------------------
PROJECT_ROOT_DIR = '.'
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, 'i')
os.makedirs(IMAGES_PATH, exist_ok=True)

#----------------------------- 2. Functions -----------------------------------
def categorical_plot(df, cat_name, **kwargs):
    """This function plots a bar plot of a categorical variable in relation to
    the binary objective variable.
    
    Inputs:
    -------
    df: Pandas DataFrame
        Dataframe with the information that will be plotted.
    cat_name: string
        Name of the categorical variable that will be analyzed.
    
    Outputs:
    --------
    fig: Matplotlib Figure
    ax: Matplotlib axe
    """
    temp_df = df.groupby([cat_name,'is_fraud']).count()['merch_long']\
        .to_frame().reset_index()
    temp_df = temp_df.pivot(
        index = cat_name,
        columns = 'is_fraud',
        values = 'merch_long'
    )
    temp_df.columns = ['Non-Fraud', 'Fraud']
    temp_df_pct = temp_df.div(temp_df.sum(axis=1), axis=0)*100
    fig, ax = plt.subplots(1,2, **kwargs)
    temp_df.plot.bar(stacked=True, ax=ax[0])
    temp_df_pct.plot.bar(stacked=True, ax=ax[1])
    ax[1].get_legend().remove()
    ax[0].set_title('Number of Non-fraud/Fraud Cases')
    ax[1].set_title('Proportion(%) of Non-fraud/Fraud Cases')
    plt.suptitle(f"Fraud Analysis by {cat_name} (# cases, % cases)", 
        fontsize=18);
    return fig, ax

def save_plot(fig_name, fig_extension='png', resolution=300):
    """Saves a plot in a images directory.
    
    Inputs:
    -------
    fig_id: string
        name/id of the saved figure
    fig_extension: string
        Determines the extension of the saved figure
    resolution: integer
        Determines the resolution of the saved image
    
    Outputs:
    --------
    None
    """
    path = os.path.join(IMAGES_PATH, fig_name+'.'+fig_extension)
    plt.savefig(path, format=fig_extension, dpi=resolution)
    print(f'Figure {fig_name} saved!')


def secrets():
    '''Contains secret information to connect to PostgreSQL server.'''
    personal_info = {
        "host": 'pg.pg4e.com',
        "port": 5432,
        "database": 'pg4e_19da1495c4',
        "user": 'pg4e_19da1495c4',
        "password": 'pg4e_p_d2d035a995f73b7'
    }
    return personal_info

def alchemy(secrets):
    '''Generates a connection string to the SQL server through AlchemySQL'''
    return f"postgresql://{secrets['user']}:{secrets['password']}@{secrets['host']}/{secrets['database']}"

#----------------------------- 3. Classes -------------------------------------
class DataFrameSelector(BaseEstimator, TransformerMixin):
    '''Object that selects certain attributes from a dataframe and returns
    its values'''
    def __init__(self, col_names):
        '''
        Inputs:
        -------
        col_names: list of strings
            list of strings to be selected.
        '''
        self.col_names = col_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.col_names].values


class PreparedData(object):
    '''Prepares dataframes according to the specification of the PayU data.'''
    def __init__(self, cat_names, num_names, target_names):
        '''
        Inputs:
        -------
        cat_names: list of strings
            Names of the categorical variables inside the inputed dataframe
        num_names: list of strings
            Names of the numerical variables inside the inputed dataframe
        target_vars: list of strings/string
            Names(s) of the objective variable.
        train: boolean
            Determines if with this dataframe the pipelines will be fitted (if
            True), or if the pipelines will just be used to transform data 
            after being fitted with training data.
        over_sample: boolean
            Determines whether it is a traning dataset or not. It it is, during
            the preprocessing phase it will perform over-sampling to the
            imbalanced class in the target variables.
        '''
        self.cat_names = cat_names
        self.num_names = num_names
        self.target_names = target_names
    
    def prepare_data(self, df, train=True, over_sample=True):
        """Prepares the full dataset so that the categorical variables are 
        one-hot encoded and joins them to th enumerical varibles. 
        Inputs:
        -------
        df: Pandas DataFrame
            Dataframe with the raw information to be transormed
        train: boolean
            Determines if with this dataframe the pipelines will be fitted (if
            True), or if the pipelines will just be used to transform data 
            after being fitted with training data.
        over_sample: boolean
            Determines whether it is a traning dataset or not. It it is, during
            the preprocessing phase it will perform over-sampling to the
            imbalanced class in the target variables.
        Outputs:
        --------
        X: NumPy array
            Array with the processed information of the predictive variables.
        Y: NumPy array
            Array with the information of the target variables.
        """
        if train:
            num_pipeline = Pipeline([
                ('selector', DataFrameSelector(self.num_names))
            ])
            cat_pipeline = Pipeline([
                ('selector', DataFrameSelector(self.cat_names)),
                ('one_hot_encoder', OneHotEncoder(sparse=False,
                                                  handle_unknown='ignore'))
            ])
            full_pipeline = FeatureUnion(transformer_list=[
                ('num_pipeline', num_pipeline),
                ('cat_pipeline', cat_pipeline)
            ])
            X = full_pipeline.fit_transform(df)
            Y = df[self.target_names].values
            self.full_pipeline = full_pipeline 

            # Over-sample the less frequent class if training:
            if over_sample:
                over_sample = SMOTE()
                X, Y = over_sample.fit_resample(X,Y)
        
        else:
            X = self.full_pipeline.transform(df)
            Y = df[self.target_names].values

        return X,np.expand_dims(Y, axis=1)


