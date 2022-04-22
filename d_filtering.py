import pandas as pd
import numpy as np

data2 = pd.read_csv("final.csv")


C = data2["vote_average"].mean()
m = data2["vote_count"].quantile(0.9)
q_movies = data2.copy().loc[data2["vote_count"]>=m]

def weight_rating(x,m=m,C=C):
  v = x["vote_count"]
  R = x["vote_average"]
  return(v/(v+m)*R)+(m/(m+v)*C)

q_movies["score"] = q_movies.apply(weight_rating, axis= 1)
q_movies = q_movies.sort_values("score",ascending = False)
output = q_movies[["original_title","poster_link","release_date","runtime","vote_average"]].head(20).values.tolist()
