import pandas as pd
import plotly.graph_objs as go
from sqlalchemy import create_engine


def load_data():
    """
    Funtion to load data from database
    
    :return df: dataframe
    """
    # load data
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql_table('Categorized_Messages', con=engine)

    Y_pred = pd.read_sql_table('Y_pred', con=engine)
    Y_test = pd.read_sql_table('Y_test', con=engine)
    
    return df, Y_test, Y_pred


def return_figures():
    """
    Creates plotly visualizations to be calles in run.py

    :return list (dict): list containing the four plotly visualizations
    """
    
    # extract data needed for visuals
    df, Y_test, Y_pred = load_data()
    
    # create visuals
    graph_one = []
    category_examples_count = df[df.columns[4:]].sum().sort_values(ascending=False)
    category_names = list(category_examples_count.index)
    graph_one.append(
        go.Bar(
            x=category_names,
            y=category_examples_count,
        )
    )
    
    layout_one = dict(title='Distribution of Message Categories',
                      xaxis=dict(title='Category'),
                      yaxis=dict(title='Count')
                      )








    graph_three = []
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    graph_three.append(
        go.Bar(
            x=genre_names,
            y=genre_counts,
        )
    )

    layout_three = dict(title='Distribution of Message Genres',
                      xaxis=dict(title='Genre'),
                      yaxis=dict(title='Count')
                      )
    
    
    
    
    
    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    #figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    
    return figures
