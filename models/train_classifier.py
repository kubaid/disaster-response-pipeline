import sys

import pandas as pd
from sqlalchemy import create_engine
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages', con=engine)
    
    df.dropna(inplace=True)
    
    X = df.loc[:, 'message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Best parameters found
    parameters = {
        'clf__estimator__min_samples_split': [2],
        'clf__estimator__n_estimators': [100],
        'vect__ngram_range': [(1, 1)]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    
    for i in range(Y_test.shape[1]):
        print(f'Classification report for category {category_names[i]}:')
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()