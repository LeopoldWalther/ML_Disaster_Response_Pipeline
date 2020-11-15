import pandas as pd
import plotly.graph_objs as go
from sqlalchemy import create_engine


def load_data():
    """
    Funtion to load data from database
    
    :return df: dataframe
    """
    # create connection to database
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    
    # load complete data
    df = pd.read_sql_table('Categorized_Messages', con=engine)

    #load model report
    df_report = pd.read_sql_table('df_report', con=engine)
    
    return df, df_report


def return_figures():
    """
    Creates plotly visualizations to be calles in run.py

    :return list (dict): list containing the four plotly visualizations
    """
    
    # extract data needed for visuals
    df, df_report = load_data()
    
    # create visualization for Distribution of Message Categories
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


    # create visualization for f1 score of trained model per category
    graph_two = []
    df_report.sort_values(by='f1_score', ascending=False, inplace=True)
    category_names_f1 = list(df_report['class'])
    category_f1_score = list(df_report.f1_score)
    graph_two.append(
        go.Bar(
            x=category_names_f1,
            y=category_f1_score,
        )
    )
    
    layout_two = dict(title='F1 Score of Model per Message Category',
                      xaxis=dict(title='Category'),
                      yaxis=dict(title='F1 Score')
                      )



    # create visualization for Distribution of Message Genres
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
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    
    return figures
