# import libraries
import pandas as pd
import numpy as np
import sqlalchemy
import re
import sys
import nltk

nltk.download(['punkt', 'wordnet', 'stopwords', 'words', 'averaged_perceptron_tagger', 'maxent_ne_chunker'])
from nltk import sent_tokenize, pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    Loads data from SQL Database and transforms it for model training
    
       Args:
       database_filepath string: SQL database file
       
       Returns:
       X array: Features dataframe
       Y array: Target dataframe
       category_names: Target labels list
       """
    
    # create engine for connection to  database
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(database_filepath))
    
    # load data from database into pandas dataframe
    df = pd.read_sql_table('Categorized_Messages', con=engine)
    
    # create feature matrix (numpy array) containing only the messages
    # create target/label matrix containing all possible categories to predict
    X, Y = df['message'], df.iloc[:, 4:]
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes text data using
    
    Args:
    text str: Messages as text data
    
    Returns:
    words list: Processed text after normalizing, tokenizing and lemmatizing
    """
    
    # Normalize
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())  # Punctuation Removal and Case Normalizing
    
    # Tokenize
    words = word_tokenize(text)
    
    # Stop Word Removal
    words = [w for w in words if w not in stopwords.words('english')]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in words]
    lemmatized = [lemmatizer.lemmatize(w, pos='v') for w in lemmatized]
    
    return lemmatized


def build_model():
    """
    Builds a model using Random Forest Classifier. Data is transformed in pipeline using Tokenization, Count Vectorizer,
    Tfidf Transformer and

    Returns:
    Trained model after performing grid search
    """
    # define pipeline with estimators including a few transformers and a classifier
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize,
                                         #ngram_range=(1, 1),
                                         #max_df=0.5,
                                         #max_features=5000
                                         )
                 ),
                ('tfidf', TfidfTransformer()),
            ]))  # ,
            # ('pos', PartOfSpeechInterpreter())
        ])),
        ('multi_clf', MultiOutputClassifier(RandomForestClassifier(
            #n_estimators=100,
            min_samples_split=4
        )))
    ])
    # Possible alternatives:  DecisionTreeClassifier, ExtraTreeClassifier, RandomForestClassifier,
    # KNeighborsClassifier, RadiusNeighborsClassifierr
    
    # define parameters to perform grid search on pipeline
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        #'features__pos__use_idf': (True, False)#,
        'multi_clf__estimator__n_estimators': [50, 100, 200],
        'multi_clf__estimator__min_samples_split': [2, 3, 4]#,
        #'features__transformer_weights': (
        #{'text_pipeline': 1, 'pos': 0.5},
        #{'text_pipeline': 0.5, 'pos': 1},
        #{'text_pipeline': 0.8, 'pos': 1},
    }

    # create GridSearchCV object with pipeline
    # n_jobs= use all processors available in parallel
    # cv= use the default 5-fold cross validation
    # verbose= more messages
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, scoring='f1_macro', cv=None, n_jobs=-1, verbose=10)

    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        
        print('Trained model saved!')
    
    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
