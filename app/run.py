import json
import plotly
from flask import Flask
from flask import render_template, request, jsonify
import joblib
from plots.figures import return_figures, load_data
from models.train_classifier import tokenize

app = Flask(__name__)

# load model
model = joblib.load("../models/classifier.pkl")
df, df_report = load_data()


# index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # create visuals
    figures = return_figures()
    
    # encode plotly graphs in JSON
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]
    
    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=figuresJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    
    # This will render the go.html Please see that file. 
    return render_template('go.html', query=query, classification_result=classification_results)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
