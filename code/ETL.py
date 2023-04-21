import numpy as np
import pandas as pd
from tqdm import tqdm


class ETL:
    def __init__(self, score_base=10):
        self.score_base = score_base
        
    
    def read_data_from_csv(self, path_review, path_movie):
        reviews = pd.read_csv(path_review)
        if path_movie is not None:
            movies = pd.read_csv(path_movie)
            return reviews, movies
        else:
            return reviews
    

    def transform_string_score(self, score: str):

        # remove white space
        while ' ' in score:
            score = score.replace(' ', '')

        score = score.upper()

        grade = ['A+', 'A', 'A-', 
                 'B+', 'B', 'B-', 
                 'C+', 'C', 'C-', 
                 'D+', 'D', 'D-', 
                 'E+', 'E', 'E-', 
                 'F+', 'F', 'F-']

        index = grade.index(score)
        return ((len(grade) - index)/len(grade))*self.score_base


    def transform_number_score(self, score: str):
        if '/' in score:
            score = score.split('/')
            base_number = float(score[1])
            if base_number == 0:
                return None
            score = float(score[0])
            return (score/base_number)*self.score_base
        else:
            return None


    def transform_score(self, score: str):
        if pd.notna(score):
            if score[0].isdigit():
                return self.transform_number_score(score)
            else:
                return self.transform_string_score(score)
        else:
            return None
    
    
    def transform_data(self, 
                       data: pd.core.frame.DataFrame, 
                       link: str,
                       sentiment_col_name: str,
                       text_col_name: str, 
                       score_col_name: str):
        # extract useful columns
        df = None
        if score_col_name is not None:
            df = data[[link, sentiment_col_name, score_col_name, text_col_name]]
        else:
            df = data[[link, sentiment_col_name, text_col_name]]

        # remove NAN in content
        df = df.dropna(subset=[text_col_name])
        # remove duplicated data and keep one of them
        df = df.drop_duplicates(subset=[text_col_name], keep='first')

        # transform scoring
        if score_col_name is not None:
            df[score_col_name] = [self.transform_score(score=s) for s in df[score_col_name]]

        return df
    
    
    def transform(self, path_review, path_movie, sentiment_col_name, text_col_name, score_col_name):
        reviews, movies = self.read_data_from_csv(path_review=path_review, path_movie=path_movie)
        
        movies = movies[['rotten_tomatoes_link', 'movie_title', 
                            'critics_consensus', 'tomatometer_rating', 
                            'tomatometer_status', 'tomatometer_count',
                             'audience_status', 'audience_rating', 
                         'audience_count']]
        
        
        reviews = self.transform_data(reviews, 
                                      link='rotten_tomatoes_link',
                                      sentiment_col_name=sentiment_col_name,
                                      text_col_name=text_col_name, 
                                      score_col_name=score_col_name)
        
        data = movies.join(reviews.set_index('rotten_tomatoes_link'), on='rotten_tomatoes_link', how='right')

        missing_movie_titles = {
            "m/-cule_valley_of_the_lost_ants": "Minuscule: Valley of the Lost Ants",
            "m/patton_oswalt_tragedy_+_comedy_equals_time": "Patton Oswalt: Tragedy Plus Comedy Equals Time",
            "m/sympathy-for-the-devil-one-+-one": "Sympathy for the Devil (One + One)",
            "m/+_one_2019": "Plus One (2019)",
            "m/+h": "Plush",
            "m/-_man": "Minus Man"
        }

        for l in missing_movie_titles:
            data.loc[data['rotten_tomatoes_link'] == l, ['movie_title']] = missing_movie_titles[l]
        
        data.reset_index(inplace=True)
        data.drop('index', axis=1, inplace=True)
        
        return data
    
    
    def save_data(self, data, path="./../../data/normalized_data.csv"):
        data.to_csv(path, index=False)