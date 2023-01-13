import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories from CSV files.
    Args:
        messages_filepath: path to the file containing messages
        categories_filepath: path to the file containing categories

    Returns:
        dataframe containing messages and categories
    """
    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge dataframes
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    """
    Apply cleaning techniques to a dataset.
    Args:
        df: dataframe containing messages and categories

    Returns:
        cleaned dataframe
    """
    # split categories into separate columns
    categories = df.categories.str.split(';', expand=True)
    # use first row of the resulting dataframe to extract column names
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the categories columns
    categories.columns = category_colnames

    # convert category values to ones and zeros
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype('str').str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        # if the numeric value is more than 1, replace it with 1
        categories[categories[column] > 1] = 1

    # replace the old category column with the new categories dataframe
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1, join='inner')

    # remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    Save dataset to a database.
    Args:
        df: dataframe
        database_filename: database file name
    """
    # create engine
    engine = create_engine(f'sqlite:///{database_filename}')
    # export dataframe
    df.to_sql('messages', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
