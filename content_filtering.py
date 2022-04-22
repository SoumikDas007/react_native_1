from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
import pandas as pd
import numpy as np



data2 = pd.read_csv("final.csv")
data2  = data2[data2['soup'].notna()]

count = CountVectorizer(stop_words='english') 
count_matrix = count.fit_transform(data2['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

data2 = data2.reset_index() 
indices = pd.Series(data2.index, index=data2['original_title'])

def get_recommendations(original_title, cosine_sim): 
  idx = indices[original_title] 
  sim_scores = list(enumerate(cosine_sim[idx])) 
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) 
  sim_scores = sim_scores[1:11] 
  movie_indices = [i[0] for i in sim_scores] 
  return data2["original_title","poster_link","release_date","runtime","vote_average","overview"].iloc[movie_indices].values.tolist()