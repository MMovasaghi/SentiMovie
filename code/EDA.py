import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import os
import pickle

nltk.download("stopwords")
nltk.download('wordnet')

class EDA:
    def __init__(self, 
                 data, 
                 text_col_name="review_content", 
                 sentiment_col_name="review_type", 
                 score_col_name="review_score"):
        
        self.data = data
        self.text_col_name = text_col_name
        self.sentiment_col_name = sentiment_col_name
        self.score_col_name = score_col_name
        self.STOPWORDS = stopwords.words("english")
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.clean_data = data.copy()
    
    
    def __len__(self):
        return len(self.data)
    
    
    def clean_text(self, text, lower=True, stem=False, lemma=False):
        # Lower
        if lower: 
            text = text.lower()

        # Remove stopwords
        if len(self.STOPWORDS):
            pattern = re.compile(r'\b(' + r"|".join(self.STOPWORDS) + r")\b\s*")
            text = pattern.sub('', text)

        # Spacing and filters
        text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)
        text = re.sub("[^A-Za-z0-9]+", " ", text)
        text = re.sub(" +", " ", text)
        text = text.strip()

        # Remove links
        text = re.sub(r"http\S+", "", text)

        # Stemming
        if stem:
            text = " ".join([self.stemmer.stem(word, to_lowercase=lower) for word in text.split(" ")])
        
        # lemmatization
        if lemma:
            text = " ".join([self.lemmatizer.lemmatize(word) for word in text.split(" ")])

        return text
    
    
    def review_number_per_movie(self, output_path="./../data/review_per_movie_dist.pkl", overwrite=False):
        if (not os.path.exists(output_path)) or overwrite:
            self.movie_titles = np.unique(list(self.movies['movie_title']))
            print("[Log] Generate \"review_number\" and save to file.")
            self.review_number = {t: len(self.movies[self.movies['movie_title'] == t]) for t in tqdm(self.movie_titles)}
            with open(output_path, 'wb') as handle:
                pickle.dump(self.review_number, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.movie_titles = np.unique(list(self.movies['movie_title']))
            print("[Log] Read \"review_number\" from file.")
            with open(output_path, 'rb') as f:
                self.review_number = pickle.load(f)
    
    
    def delete_outlier_movie(self, data, thr_tomatometer_count=5, thr_audience_count=100):
        self.movies = data[(data['tomatometer_count'] >= thr_tomatometer_count) &  
                     (data['audience_count'] >= thr_audience_count)].copy()
        self.movies = self.movies[self.movies['movie_title'].notna()]
        self.movies.reset_index(inplace=True)
        self.movies.drop('index', axis=1, inplace=True)
    
    
    def delete_outlier_movie_based_review_number(self, thr=10):
        if self.review_number:
            df = {"movie_title": list(self.review_number.keys()), 
                  "review_number": list(self.review_number.values())}
            df = pd.DataFrame(df)
            df = df[df['review_number'] >= thr]
            self.movies = df.join(self.movies.set_index('movie_title'), on='movie_title', how='inner')
            self.movies.reset_index(inplace=True)
            self.movies.drop(['review_number', 'index'], axis=1, inplace=True)
            return self.movies
        else:
            raise Exception("[Error] First, the amount of the \"review_number\" should be calculated.")
    
    
    def cleanning_data(self, stem=False, lemma=False):
        self.clean_data[self.text_col_name] = [self.clean_text(t, stem=stem, lemma=lemma) for t in self.data[self.text_col_name]]
    
    
    def get_scores(self, data, review_type=None):
        scores = None
        if review_type is not None:
            scores = list(data[data[self.sentiment_col_name] == review_type][self.score_col_name])
        else:
            scores = list(data[self.score_col_name])
        scores = [x for x in scores if x == x]
        return np.sort(scores)
    
    
    def number_of_na(self, data, col):
        return data[col].isna().sum()
    
    
    def get_col_value(self, data, col, value):
        return data[data[col] == value]
    
    
    def class_numbers(self, data, labels=['Fresh', 'Rotten']):
        label = {}
        label[labels[0]] = len(self.get_col_value(data=data, col=self.sentiment_col_name, value=labels[0]))
        label[labels[1]] = len(self.get_col_value(data=data, col=self.sentiment_col_name, value=labels[1]))
        return label
        
    
    def get_statistics(self, data, review_type=None):
        scores = self.get_scores(data, review_type=review_type)
        texts = None
        if review_type is not None:
            texts = list(data[data[self.sentiment_col_name] == review_type][self.text_col_name])
        else:
            texts = list(data[self.text_col_name])
        texts_len = [len(t) for t in texts]
        statistics = {
            "len": len(scores),
            "score_mean": np.mean(scores),
            "score_median": np.median(scores),
            "score_std": np.std(scores),
            "score_min": np.min(scores),
            "score_max": np.max(scores),
            "texts_len_mean": np.mean(texts_len),
            "texts_len_median": np.median(texts_len),
            "texts_len_std": np.std(texts_len),
            "texts_len_min": np.min(texts_len),
            "texts_len_max": np.max(texts_len),
        }
        if review_type is None:
            label = self.class_numbers(data)
            for l in label:
                statistics[f'{l}_number'] = label[l]
        return statistics
    
    
    def get_movie_dataset(self, data):
        return data[['rotten_tomatoes_link', 'movie_title', 'tomatometer_rating', 
                    'audience_rating', 'review_score', 'review_type', 'review_content']]
    
    
    def statistics_of_data(self, data):
        all_data = self.get_statistics(data)
        fresh_data = self.get_statistics(data, review_type="Fresh")
        rotten_data = self.get_statistics(data, review_type="Rotten")
        return all_data, fresh_data, rotten_data
    
    
    def save_data(self, data, path="./../../data/normalized_data.csv"):
        data.to_csv(path, index=False)
    