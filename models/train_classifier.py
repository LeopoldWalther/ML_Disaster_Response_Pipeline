# import libraries
import pandas as pd
import numpy as np
import sqlalchemy
import re
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import pickle
import warnings
warnings.filterwarnings('ignore')
nltk.download(['punkt', 'wordnet', 'stopwords', 'words'])


def load_data(database_filepath):
    """
    Loads data from SQL Database and transforms it for model training
    
    :param:
        database_filepath: SQL database file (string)
    :returns:
        x: Features (dataframe)
        y: Targets (dataframe)
        category_names: Target labels (list)
    """
    
    # create engine for connection to  database
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(database_filepath))
    
    # load data from database into pandas dataframe
    df = pd.read_sql_table('Categorized_Messages', con=engine)
    
    # create feature matrix (numpy array) containing only the messages
    X = df['message']
    # create target/label matrix containing all possible categories to predict
    Y = df.iloc[:, 4:]
    category_names = df.iloc[:, 4:].columns
    
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
    stop_words = stopwords.words('english')
    words = [w for w in words if w not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lem = [lemmatizer.lemmatize(w) for w in words]
    lem = [lemmatizer.lemmatize(w, pos='v') for w in lem]
    
    return lem


def build_model():
    """
    Builds a model using Random Forest Classifier. Data is transformed in pipeline using Tokenization, Count Vectorizer,
    Tfidf Transformer and
    
    :return: cv: Trained model after performing grid search (GridSearchCV model)
    """
    
    # define pipeline with estimators including a few transformers and a classifier
    pipeline = Pipeline([
        
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 2), max_df=0.5, max_features=None)),
            ('tfidf', TfidfTransformer()),
        ])),
        ('multi_clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
    ])
    
    # define parameters to perform grid search on pipeline
    parameters = {
        #'text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'text_pipeline__vect__max_features': (None, 10000)
    }
    
    # create GridSearchCV object with pipeline
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, scoring='f1_macro', n_jobs=1, verbose=10)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Measures model's performance on test data and prints out results.
    
    :param model: trained model (GridSearchCV Object)
    :param X_test: Test features (dataframe)
    :param Y_test: Test targets (dataframe)
    :param category_names: Target labels (dataframe)
    """
    
    # predict target values Y_pred of test features
    Y_pred = model.predict(X_test)

    print(classification_report(Y_test, Y_pred, target_names=category_names))
    
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    """
    Function to save trained model as pickle file.

    :param model: Trained model (GridSearchCV Object)
    :param model_filepath: Filepath to store model (string)

    :return: None
    """
    # save model
    pickle.dump(model, open(model_filepath, 'wb'))

    
    
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
