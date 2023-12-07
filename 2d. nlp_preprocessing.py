import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import string
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import copy
import sys 
import os 
import pickle 

from argparse import ArgumentParser
from gensim.models import KeyedVectors
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

from nltk import sent_tokenize
from nltk import pos_tag
from nltk import map_tag
from nltk import word_tokenize
from nltk.corpus import stopwords


# Load NLTK's English stop-words list
stop_words = set(stopwords.words('english'))

def word_grams(text, min=1, max=4):
    '''
    Function to create N-grams from text
    Required Input -
        - text = text string for which N-gram needs to be created
        - min = minimum number of N
        - max = maximum number of N
    Expected Output -
        - s = list of N-grams 
    '''
    s = []
    for n in range(min, max+1):
        for ngram in ngrams(text, n):
            s.append(' '.join(str(i) for i in ngram))
    return s


def generate_bigrams_df(df, column_names):
    """
    Generate bigrams from specified columns in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame to generate bigrams from.
    column_names (list of str): List of column names to generate bigrams from.

    Returns:
    pd.DataFrame: DataFrame with bigrams appended as new columns.
    """
    bigram_columns = []
    for col in column_names:
        bigram_col = f"{col}_bigrams"
        bigram_columns.append(bigram_col)
        df[bigram_col] = df[col].apply(lambda x: generate_bigrams([x]))
    return df[bigram_columns]

def make_wordcloud(df,column, bg_color='white', w=1200, h=1000, font_size_max=50, n_words=40,g_min=1,g_max=1):
    '''
    Function to make wordcloud from a text corpus
    Required Input -
        - df = Pandas DataFrame
        - column = name of column containing text
        - bg_color = Background color
        - w = width
        - h = height
        - font_size_max = maximum font size allowed
        - n_word = maximum words allowed
        - g_min = minimum n-grams
        - g_max = maximum n-grams
    Expected Output -
        - World cloud image
    '''
    text = ""
    for ind, row in df.iterrows(): 
        text += row[column] + " "
    text = text.strip().split(' ') 
    text = word_grams(text,g_min,g_max)
    
    text = list(pd.Series(word_grams(text,1,2)).apply(lambda x: x.replace(' ','_')))
    
    s = ""
    for i in range(len(text)):
        s += text[i] + " "

    wordcloud = WordCloud(background_color=bg_color, \
                          width=w, \
                          height=h, \
                          max_font_size=font_size_max, \
                          max_words=n_words).generate(s)
    wordcloud.recolor(random_state=1)
    plt.rcParams['figure.figsize'] = (20.0, 10.0)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
def generate_wordcloud(df, column_names):
    """
    Generates a wordcloud from a pandas DataFrame

    Parameters:
    df (pd.DataFrame): DataFrame containing the data
    column_names (list): List of column names in the DataFrame to generate the wordcloud from

    Returns:
    None
    """
    all_words = ' '.join([' '.join(text) for col in column_names for text in df[col]])
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()
    
    
def get_tokens(text):
    '''
    Function to tokenize the text
    Required Input - 
        - text - text string which needs to be tokenized
    Expected Output -
        - text - tokenized list output
    '''
    return word_tokenize(text)

def tokenize_columns(dataframe, columns):
    """
    Tokenize the values in specified columns of a pandas DataFrame.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame to tokenize.
        columns (list): A list of column names to tokenize.

    Returns:
        pandas.DataFrame: A new DataFrame with tokenized values in the specified columns.
    """
    # Download necessary NLTK resources if they haven't been downloaded yet
    nltk.download('punkt')

    # Create a new DataFrame to hold the tokenized values
    tokenized_df = pd.DataFrame()

    # Tokenize the values in each specified column
    for col in columns:
        # Tokenize the values in the current column using NLTK's word_tokenize function
        tokenized_values = dataframe[col].apply(nltk.word_tokenize)

        # Add the tokenized values to the new DataFrame
        tokenized_df[col] = tokenized_values

    # Return the new DataFrame with tokenized values
    return tokenized_df

#another way
# --------------------------------------------------------------------------
def tokenize(text, sep=' ', preserve_case=False):
    """
    Tokenize a string into a list of tokens.

    Parameters:
    text (str): String to be tokenized
    sep (str, optional): Separator to use for tokenization. Defaults to ' '.
    preserve_case (bool, optional): Whether to preserve the case of the text. Defaults to False.

    Returns:
    list: List of tokens
    """
    if not preserve_case:
        text = text.lower()
    tokens = text.split(sep)
    return tokens

def tokenize_df(df, column_names, sep=' ', preserve_case=False):
    """
    Tokenize a pandas dataframe with multiple columns.

    Parameters:
    df (pd.DataFrame): Dataframe to be tokenized
    columns (list of str): List of column names to be tokenized
    sep (str, optional): Separator to use for tokenization. Defaults to ' '.
    preserve_case (bool, optional): Whether to preserve the case of the text. Defaults to False.

    Returns:
    pd.DataFrame: Tokenized dataframe
    """
    for col in column_names:
        df[col] = df[col].apply(lambda x: tokenize(x, sep, preserve_case))
    return df

carbon_google1 = tokenize_df (carbon_google1, column_names =  ["title"], sep=' ', preserve_case=False)
# --------------------------------------------------------------------------------------------------------------
def bag_of_words_features(df, text_columns, target_columns):
    """
    This function takes in a DataFrame and one or two columns and returns a bag of words representation of the data as a DataFrame.

    Parameters:
    df (pandas DataFrame): The DataFrame to extract features from.
    column1 (str): The name of the first column to use as input data.
    column2 (str, optional): The name of the second column to use as input data. If not provided, only the first column will be used.

    Returns:
    pandas DataFrame: The bag of words representation of the input data as a DataFrame.
    """
        
    text_data = df[text_columns].apply(lambda x: " ".join([str(i) for i in x]), axis=1)

    text_data = text_data.str.lower()
    vectorizer = CountVectorizer(max_df=0.90, min_df=4, max_features=1000, stop_words=None)
    X_bow = vectorizer.fit_transform(text_data)
    # Use the new function to get the feature names
    feature_names = vectorizer.get_feature_names_out()
    df.dropna(subset=[target_column], inplace=True) if target_columns else None

    X_bow = pd.DataFrame(X_bow.toarray(), columns=feature_names)
    
    if target_columns:        
        y = df[target_columns]
        return X_bow, y
    
    return X_bow

def convert_lowercase(text):
    '''
    Function to tokenize the text
    Required Input - 
        - text - text string which needs to be lowercased
    Expected Output -
        - text - lower cased text string output
    '''
    return text.lower()

def remove_unwanted_characters(df, columns):        #clean text A
    """
    Remove unwanted characters (including smileys and emojies) from specified columns in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): A list of column names to clean.
    unwanted_chars (str): The characters to remove.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    import re 
    unwanted_chars = '[$#&*@%]'
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\u2764\ufe0f" # heart emoji
                           "]+", flags=re.UNICODE)
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: emoji_pattern.sub(r'', x))
            df[col] = df[col].str.replace(unwanted_chars, '')
        else:
            print(f"Column '{col}' does not exist in the DataFrame.")
    return df

def remove_punctuations(text):
    '''
    Function to tokenize the text
    Required Input - 
        - text - text string 
    Expected Output -
        - text - text string with punctuation removed
    '''
    return text.translate(None,string.punctuation)

def remove_stopwords(text):
    '''
    Function to tokenize the text
    Required Input - 
        - text - text string which needs to be tokenized
    Expected Output -
        - text - list output with stopwords removed
    '''
    return [word for word in text.split() if word not in stop_words]

def remove_short_words(df, column_names, min_length=3):
    """Remove short words from columns in a pandas DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to modify.
    column_names (List[str]): A list of column names to modify.
    min_length (int, optional): The minimum length of words to keep. Default is 3.

    Returns:
    pandas.DataFrame: The modified DataFrame with short words removed from specified columns.
    """
    for column_name in column_names:
        df[column_name] = df[column_name].apply(
            lambda x: ' '.join([word for word in x.split() if len(word) >= min_length])
        )
    return df

def convert_stemmer(word):
    '''
    Function to tokenize the text
    Required Input - 
        - word - word which needs to be tokenized
    Expected Output -
        - text - word output after stemming
    '''
    porter_stemmer = PorterStemmer()
    return porter_stemmer.stem(word)

def stem_df(df, column_names):
    """
    Perform stemming on a pandas dataframe with multiple columns.

    Parameters:
    df (pd.DataFrame): Dataframe to be stemmed
    columns (list of str): List of column names to be stemmed

    Returns:
    pd.DataFrame: Stemmed dataframe
    """
    stemmer = PorterStemmer()
    for col in column_names:
        df[col] = df[col].apply(lambda x: [stemmer.stem(i) for i in x])
    return df

def convert_lemmatizer(word):
    '''
    Function to tokenize the text
    Required Input - 
        - word - word which needs to be lemmatized
    Expected Output -
        - word - word output after lemmatizing
    '''
    wordnet_lemmatizer = WordNetLemmatizer()
    return wordnet_lemmatizer.lemmatize(word)
    
def create_tf_idf(df, column, train_df = None, test_df = None,n_features = None):
    '''
    Function to do tf-idf on a pandas dataframe
    Required Input -
        - df = Pandas DataFrame
        - column = name of column containing text
        - train_df(optional) = Train DataFrame
        - test_df(optional) = Test DataFrame
        - n_features(optional) = Maximum number of features needed
    Expected Output -
        - train_tfidf = train tf-idf sparse matrix output
        - test_tfidf = test tf-idf sparse matrix output
        - tfidf_obj = tf-idf model
    '''
    tfidf_obj = TfidfVectorizer(ngram_range=(1,1), stop_words='english', 
                                analyzer='word', max_features = n_features)
    tfidf_text = tfidf_obj.fit_transform(df.ix[:,column].values)
    
    if train_df is not None:        
        train_tfidf = tfidf_obj.transform(train_df.ix[:,column].values)
    else:
        train_tfidf = tfidf_text

    test_tfidf = None
    if test_df is not None:
        test_tfidf = tfidf_obj.transform(test_df.ix[:,column].values)

    return train_tfidf, test_tfidf, tfidf_obj
    
def create_countvector(df, column, train_df = None, test_df = None,n_features = None):
    '''
    Function to do count vectorizer on a pandas dataframe
    Required Input -
        - df = Pandas DataFrame
        - column = name of column containing text
        - train_df(optional) = Train DataFrame
        - test_df(optional) = Test DataFrame
        - n_features(optional) = Maximum number of features needed
    Expected Output -
        - train_cvect = train count vectorized sparse matrix output
        - test_cvect = test count vectorized sparse matrix output
        - cvect_obj = count vectorized model
    '''
    cvect_obj = CountVectorizer(ngram_range=(1,1), stop_words='english', 
                                analyzer='word', max_features = n_features)
    cvect_text = cvect_obj.fit_transform(df.ix[:,column].values)
    
    if train_df is not None:
        train_cvect = cvect_obj.transform(train_df.ix[:,column].values)
    else:
        train_cvect = cvect_text
        
    test_cvect = None
    if test_df is not None:
        test_cvect = cvect_obj.transform(test_df.ix[:,column].values)

    return train_cvect, test_cvect, cvect_obj


def clean_text(text, full_clean=False, punctuation=False, numbers=False, lower=False, extra_spaces=False,
               control_characters=False, tokenize_whitespace=False, remove_characters=''):
    r"""
    Clean text using various techniques.

    I took inspiration from the cleantext library `https://github.com/prasanthg3/cleantext`. I did not like the whole
    implementation so I made my own changes.

    Note:
        As in the original cleantext library I will add: stop words removal, stemming and
        negative-positive words removal.

    Arguments:

        text (:obj:`str`):
            String that needs cleaning.

        full_clean (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Remove: punctuation, numbers, extra space, control characters and lower case. This argument is optional and
            it has a default value attributed inside the function.

        punctuation (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Remove punctuation from text. This argument is optional and it has a default value attributed inside
            the function.

        numbers (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Remove digits from text. This argument is optional and it has a default value attributed inside
            the function.

        lower (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Lower case all text. This argument is optional and it has a default value attributed inside the function.

        extra_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Remove extra spaces - everything beyond one space. This argument is optional and it has a default value
            attributed inside the function.

        control_characters (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Remove characters like `\n`, `\t` etc.This argument is optional and it has a default value attributed
            inside the function.

        tokenize_whitespace (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Return a list of tokens split on whitespace. This argument is optional and it has a default value
            attributed inside the function.

        remove_characters (:obj:`str`, `optional`, defaults to :obj:`''`):
            Remove defined characters form text. This argument is optional and it has a default value attributed
            inside the function.

    Returns:

        :obj:`str`: Clean string.

    Raises:

        ValueError: If `text` is not of type string.

        ValueError: If `remove_characters` needs to be a string.

    """

    if not isinstance(text, str):
        # `text` is not type of string
        raise ValueError("`text` is not of type str!")

    if not isinstance(remove_characters, str):
        # remove characters need to be a string
        raise ValueError("`remove_characters` needs to be a string!")

    # all control characters like `\t` `\n` `\r` etc.
    # Stack Overflow: https://stackoverflow.com/a/8115378/11281368
    control_characters_list = ''.join([chr(char) for char in range(1, 32)])

    # define control characters table
    table_control_characters = str.maketrans(dict.fromkeys(control_characters_list))

    # remove punctuation table
    table_punctuation = str.maketrans(dict.fromkeys(string.punctuation))

    # remove numbers table
    table_digits = str.maketrans(dict.fromkeys('0123456789'))

    # remove certain characters table
    table_remove_characters = str.maketrans(dict.fromkeys(remove_characters))

    # make a copy of text to make sure it doesn't affect original text
    cleaned = copy.deepcopy(text)

    if full_clean or punctuation:
        # remove punctuation
        cleaned = cleaned.translate(table_punctuation)

    if full_clean or numbers:
        # remove numbers
        cleaned = cleaned.translate(table_digits)

    if full_clean or extra_spaces:
        # remove extra spaces - also removes control characters
        # Stack Overflow https://stackoverflow.com/a/2077906/11281368
        cleaned = re.sub('\s+', ' ', cleaned).strip()

    if full_clean or lower:
        # lowercase
        cleaned = cleaned.lower()

    if control_characters:
        # remove control characters
        cleaned = cleaned.translate(table_control_characters)

    if tokenize_whitespace:
        # tokenizes text n whitespace
        cleaned = re.split('\s+', cleaned)

    if remove_characters:
        # remove these characters from text
        cleaned = cleaned.translate(table_remove_characters)

    return cleaned


def preprocess_text(text):
    text = re.sub(r"http\S+", "", text) # remove URLs
    text = re.sub('@[^\s]+', '', text) # remove usernames
    text = re.sub('#', '', text) # remove hashtags
    text = re.sub(r'\d+', '', text) # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    text = text.lower() # convert to lowercase
    return text


def tag_pos(x):
    sentences = sent_tokenize(x.decode("utf8"))
    sents = []
    for s in sentences:
        text = word_tokenize(s)
        pos_tagged = pos_tag(text)
        simplified_tags = [
            (word, map_tag('en-ptb', 'universal', tag)) for word, tag in pos_tagged]
        sents.append(simplified_tags)
    return sents


def post_tag_documents(data_df):
    x_data = []
    y_data = []
    total = len(data_df['plot'].as_matrix().tolist())
    plots = data_df['plot'].as_matrix().tolist()
    genres = data_df.drop(['plot', 'title', 'plot_lang'], axis=1).as_matrix()
    for i in range(len(plots)):
        sents = tag_pos(plots[i])
        x_data.append(sents)
        y_data.append(genres[i])
        i += 1
        if i % 5000 == 0:
            print (i, "/", total)

    return x_data, y_data


def word2vec(x_data, pos_filter):

    print ("Loading GoogleNews-vectors-negative300.bin")
    google_vecs = KeyedVectors.load_word2vec_format(
        'GoogleNews-vectors-negative300.bin', binary=True, limit=200000)

    print ("Considering only", pos_filter)
    print ("Averaging Word Embeddings...")
    x_data_embeddings = []
    total = len(x_data)
    processed = 0
    for tagged_plot in x_data:
        count = 0
        doc_vector = np.zeros(300)
        for sentence in tagged_plot:
            for tagged_word in sentence:
                if tagged_word[1] in pos_filter:
                    try:
                        doc_vector += google_vecs[tagged_word[0]]
                        count += 1
                    except KeyError:
                        continue

        doc_vector /= count
        if np.isnan(np.min(doc_vector)):
            continue

        x_data_embeddings.append(doc_vector)

        processed += 1
        if processed % 10000 == 0:
            print (processed, "/", total)

    return np.array(x_data_embeddings)


def doc2vec(data_df):
    data = []
    print ("Building TaggedDocuments")
    total = len(data_df[['title', 'plot']].as_matrix().tolist())
    processed = 0
    for x in data_df[['title', 'plot']].as_matrix().tolist():
        label = ["_".join(x[0].split())]
        words = []
        sentences = sent_tokenize(x[1].decode("utf8"))
        for s in sentences:
            words.extend([x.lower() for x in word_tokenize(s)])
        doc = TaggedDocument(words, label)
        data.append(doc)

        processed += 1
        if processed % 10000 == 0:
            print (processed, "/", total)

    model = Doc2Vec(min_count=1, window=10, size=300, sample=1e-4, negative=5, workers=2)
    print ("Building Vocabulary")
    model.build_vocab(data)

    for epoch in range(20):
        print ("Training epoch %s" % epoch)
        model.train(data)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
        model.train(data)

    # Build doc2vec vectors
    x_data = []
    y_data = []
    genres = data_df.drop(['title', 'plot', 'plot_lang'], axis=1).as_matrix()
    names = data_df[['title']].as_matrix().tolist()
    for i in range(len(names)):
        name = names[i][0]
        label = "_".join(name.split())
        x_data.append(model.docvecs[label])
        y_data.append(genres[i])

    return np.array(x_data), np.array(y_data)


#
# train classifiers and argument handling
#

def train_test_svm(x_data, y_data, genres):

    stratified_split = StratifiedShuffleSplit(n_splits=2, test_size=0.33)
    for train_index, test_index in stratified_split.split(x_data, y_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

    """
    print "LinearSVC"
    pipeline = Pipeline([
        ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
    ])
    parameters = {
        "clf__estimator__C": [0.01, 0.1, 1],
        "clf__estimator__class_weight": ['balanced', None],
    }
    grid_search(x_train, y_train, x_test, y_test, genres, parameters, pipeline)

    print "LogisticRegression"
    pipeline = Pipeline([
        ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
    ])
    parameters = {
        "clf__estimator__C": [0.01, 0.1, 1],
        "clf__estimator__class_weight": ['balanced', None],
    }
    grid_search(x_train, y_train, x_test, y_test, genres, parameters, pipeline)
    """

    print ("LinearSVC")
    pipeline = Pipeline([
        ('clf', OneVsRestClassifier(SVC(), n_jobs=1)),
    ])
    """
    parameters = {
        "clf__estimator__C": [0.01, 0.1, 1],
        "clf__estimator__class_weight": ['balanced', None],
    }
    """
    parameters = [

        {'clf__estimator__kernel': ['rbf'],
         'clf__estimator__gamma': [1e-3, 1e-4],
         'clf__estimator__C': [1, 10]
        },

        {'clf__estimator__kernel': ['poly'],
         'clf__estimator__C': [1, 10]
        }
         ]

    grid_search(x_train, y_train, x_test, y_test, genres, parameters, pipeline)


def grid_search(train_x, train_y, test_x, test_y, genres, parameters, pipeline):
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=10)
    grid_search_tune.fit(train_x, train_y)

    print
    print("Best parameters set:")
    print (grid_search_tune.best_estimator_.steps)
    print

    # measuring performance on test set
    print ("Applying best classifier on test data:")
    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(test_x)

    print (classification_report(test_y, predictions, target_names=genres))


def parse_arguments():
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        '--clf', dest='classifier', choices=['nb', 'linearSVC', 'logit'])

    arg_parser.add_argument(
        '--vectors', dest='vectors', type=str, choices=['tfidf', 'word2vec', 'doc2vec'])

    return arg_parser, arg_parser.parse_args()




