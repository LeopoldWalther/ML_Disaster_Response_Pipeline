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
    
    return df


def return_figures():
    """
    Creates plotly visualizations to be calles in run.py

    :return list (dict): list containing the four plotly visualizations
    """
    
    # extract data needed for visuals
    df = load_data()
    
    # create visuals
    graph_one = []
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    graph_one.append(
        go.Bar(
            x=genre_names,
            y=genre_counts,
        )
    )
    
    layout_one = dict(title='Distribution of Message Genres',
                      xaxis=dict(title='Genre'),
                      yaxis=dict(title='Count')
                      )
    
    graph_two = []
    category_examples_count = df[df.columns[4:]].sum().sort_values(ascending=False)
    category_names = list(category_examples_count.index)
    graph_two.append(
        go.Bar(
            x=category_names,
            y=category_examples_count,
        )
    )
    
    layout_two = dict(title='Distribution of Message Categories',
                      xaxis=dict(title='Category'),
                      yaxis=dict(title='Count')
                      )
    
    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append((dict(data=graph_two, layout=layout_two)))
    
    return figures
