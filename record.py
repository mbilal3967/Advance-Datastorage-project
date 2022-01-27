import pandas as pd
import sqlalchemy as db
import streamlit as st
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import Column, Integer, Float, Date, String, VARCHAR, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.query import Query
from sqlalchemy.sql.compiler import StrSQLCompiler
import os
if not os.path.isdir('ml-25m'):
    print("Please download and extact the movieLens data in current working dir using Link http://files.grouplens.org/datasets/movielens/ml-25m.zip")
if not os.path.isdir('ml-100k'):
    print("Please download and extact the movieLens data in current working dir using Link http://files.grouplens.org/datasets/movielens/ml-100k.zip")

DATABASE_USER = "root"
DATABASE_PASSWD = "786786"
DATABASE_HOST = "localhost"
DATABASE_NAME = "db1"
# DATABASE_PORT = "32000"

# makeing connection to database.
# database_connection_str = "mysql+mysqldb://%s:%s@%s:%s/%s" %(DATABASE_USER,DATABASE_PASSWD,DATABASE_HOST,DATABASE_PORT,DATABASE_NAME)
database_connection_str = "mysql+pymysql://%s:%s@%s/%s" %(DATABASE_USER,DATABASE_PASSWD,DATABASE_HOST,DATABASE_NAME)
# database_connection_str = "mysql+pymysql://root:786786@localhost/db1"
@st.cache(allow_output_mutation=True)
def get_connection():
    return db.create_engine(database_connection_str, pool_timeout=20, pool_recycle=299)
# engine = db.create_engine(database_connection_str)
# engine = db.create_engine(database_connection_str, fast_executemany = True)
# connection = engine.connect()
Session = sessionmaker(bind=get_connection())
metadata = db.MetaData()
Base=declarative_base()
session = Session()


class Movies(Base):
    __tablename__ = 'movies'
    movieId = Column(Integer, primary_key=True)
    title = Column(String)
    genres = Column(String)
    
class Ratings(Base):
    __tablename__ = 'ratings'
    ratingId = Column(Integer, primary_key=True)
    userId = Column(Integer)
    movieId = Column(Integer, ForeignKey('Movies.movieId'))
    rating = Column(Float)
    timestamp = Column(Integer)
    
class RatingsAverageCount(Base):
    __tablename__ = 'ratings_average_count'
    movieId = Column(Integer, ForeignKey('Movies.movieId'), primary_key=True)
    average = Column(Float)
    count = Column(Integer)
    
class Tags(Base):
    __tablename__ = 'tags'
    tagId = Column(Integer, primary_key=True)
    userId = Column(Integer, ForeignKey('Ratings.userId'))
    movieID = Column(Integer, ForeignKey('Movies.movieId'))
    tag = Column(String)
    timestamp = Column(Integer)

class RatingsMergedSmall(Base):
    __tablename__ = 'ratings_merged_small'
    ratingId = Column(Integer, primary_key=True)
    userId = Column(Integer)
    movieId = Column(Integer)
    rating = Column(Float)
    timestamp = Column(Integer)
    title = Column(String)
    
def serialize_orm_query(q: Query):
    """Build SQL statement with inlined parameters
    https://stackoverflow.com/questions/5631078/sqlalchemy-print-the-actual-query
    """
    return q.statement.compile(compile_kwargs={"literal_binds": True}).string


@st.cache(
    hash_funcs={
        Query: serialize_orm_query,  # if we get a query, stringify it with inbound params
        StrSQLCompiler: lambda _: None,  # we don't really care about the compiler
    },
    suppress_st_warning=True, allow_output_mutation=True
)
def orm_query(q: Query):
    return pd.read_sql_query(sql=q.statement, con=get_connection())

def Insert_movies_table(movie_info):
    movie_info.to_sql(name=Movies.__tablename__, con=get_connection(), index=False, if_exists='replace')
    print("-------------------------Movies Inserted-----------------------------")

    
def Insert_ratings_table(movie_ratings):
    print(movie_ratings.head(20))
    len_df_rating=len(movie_ratings)
    for i in range(0,len_df_rating,1000000):
        connection = get_connection().connect()
        session = Session()
        if i>len_df_rating-1000000:
            j=len_df_rating
        else:
            j=i+1000000
        (movie_ratings.iloc[i:j]).to_sql(name=Ratings.__tablename__, con=get_connection(), index=True, index_label="ratingId", if_exists='append')
        print(i,j)
        print("-------------------------1M Ratings Inserted-----------------------------")
        # print(df_ratings.iloc[[i]])
        session.commit()
        connection.close()

def Insert_average_count_ratings(df_ratings):
    # st.write("## Search for the best rated films by genre")
    mean_rated_movies_dict = df_ratings.groupby(["movieId"]).mean()
    mean_rated_movies_dict.rename(columns={'rating': 'average'}, inplace=True)
    count_rated_movies_dict = df_ratings.groupby(["movieId"]).count()
    count_rated_movies_dict.rename(columns={'rating': 'count'}, inplace=True)
    average_count_ratings = pd.merge(mean_rated_movies_dict.loc[mean_rated_movies_dict["average"]>=4], count_rated_movies_dict.loc[count_rated_movies_dict["count"]>500], on='movieId')
    print(average_count_ratings.head(5))
    average_count_ratings.to_sql(name=RatingsAverageCount.__tablename__, con=get_connection(), index=True, if_exists='replace')
    print("-------------------------Average and counts of ratings Inserted-----------------------------")
    
def Insert_ratings_merged_small_table(ratings_merged):
    ratings_merged.to_sql(name=RatingsMergedSmall.__tablename__, con=get_connection(), index=True, if_exists='replace')
    print("-------------------------ratings_merged Inserted-----------------------------")
    
def get_ratings_merged_small():
    print("Getting rating from database")
    ratings_merged_s = session.query(RatingsMergedSmall.userId, RatingsMergedSmall.movieId, RatingsMergedSmall.rating, RatingsMergedSmall.timestamp, RatingsMergedSmall.title)
    df_rat = orm_query(ratings_merged_s)
    # print(df_rat)
    print("ratings_merged_small are fetched")
    return df_rat
    
def get_knn_dataset_large(df_movies, df_ratings):
    genres_dummies =df_movies['genres'].str.get_dummies(sep='|')
    genres_dummies.index.name = "index"
    # print(genres_dummies.head())
    df_movies.index.name = "index"
    # print(df_movies.index.name)
    # print(list(df_movies.columns))
    movies_genres_dataset = pd.merge(df_movies, genres_dummies, how='inner', on='index')
    movies_genres_dataset.set_index("movieId", inplace=True)
    # print(movies_genres_dataset.head())
    # # print(list(prepared_dataset.columns))
    
    prepared_dataset = pd.merge(df_ratings, movies_genres_dataset.drop(columns=['genres', 'title']), how='inner', on='movieId')
    prepared_dataset.drop(columns=['timestamp'])
    print(prepared_dataset.head())
    # st.dataframe(prepared_dataset.head())
    # knn_dataset = prepared_dataset.groupby(by=['userId','movieId'], as_index=False).agg({"rating":"mean"})
    # # knn_dataset = prepared_dataset.groupby(['userId'])('movieId').mean()
    # knn_dataset.head()
    # print(knn_dataset.head(5))
    # #
    # userid__movieid_df = knn_dataset.pivot_table(index='userId',columns='movieId',values='rating').fillna(0)
    # #
    # # userid__movieid_df = knn_dataset.pivot( 
    # #     index='userId',
    # #     columns='movieId',
    # #     values='rating').fillna(0)
    # print(userid__movieid_df.head())
    # # average_count_ratings.to_sql(name=RatingsAverageCount.__tablename__, con=get_connection(), index=True, if_exists='replace')
    
@st.cache(allow_output_mutation=True)
def get_knn_dataset(prepared_dataset):
    # df_ratings = pd.read_csv('ml-100k/u.data', sep='\t',header=None,names=['userId','movieId','rating','timestamp'])
    # # print(df_ratings.head())
    # df_movies = pd.read_csv('ml-100k/u.item', sep='|',header=None,names='movieId | title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western'.split(' | ') ,encoding='latin-1')
    # # print(items_dataset.head())
    # prepared_dataset = pd.merge(df_ratings, df_movies[['movieId','title']], how='inner', on='movieId')
    # print(prepared_dataset.head())
    knn_dataset = prepared_dataset.groupby(by=['userId','movieId'], as_index=False).agg({"rating":"mean"})
    # print(knn_dataset.head(5))
    userid_movieid_df = knn_dataset.pivot(
        index='userId',
        columns='movieId',
        values='rating').fillna(0)
    # transform matrix to scipy sparse matrix
    userid_movieid_sm = csr_matrix(userid_movieid_df.values)
    # userid_movieid_sm
    # print(userid_movieid_sm)
    # Now we will be fitting K-Nearest Neighbours model using sparse matrix:
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(userid_movieid_sm)
    print("-------------------------KNN calculated-----------------------------")
    return model, userid_movieid_df, prepared_dataset

    
def Insert_tags_table(movie_tags):
    movie_tags.to_sql(name=Tags.__tablename__, con=get_connection(), index=True, index_label="tagId", if_exists='replace')
    print("-------------------------Tags Inserted-----------------------------")
    
@st.cache(allow_output_mutation=True)
def get_similar_users(knn_model,user_to_movie_df,user, n = 5):
  knn_input = np.asarray([user_to_movie_df.values[user-1]])
  distances, indices = knn_model.kneighbors(knn_input, n_neighbors=n+1)
  return indices.flatten()[1:] + 1, distances.flatten()[1:]


@st.cache(persist=True)
def paginate_dataframe(dataframe, page_size, page_num):
    
    page_size = page_size

    if page_size is None:

        return None

    offset = page_size*(page_num-1)

    return dataframe[offset:offset + page_size]


engine = get_connection()
insp = db.inspect(engine)
# print(insp.has_table("movies", schema="db1"))
if not insp.has_table("movies", schema="db1"):
    df_movies = pd.read_csv("ml-25m/movies.csv")
    Insert_movies_table(df_movies)

if not insp.has_table("ratings", schema="db1"):
    df_ratings = pd.read_csv("ml-25m/ratings.csv")
    Insert_ratings_table(df_ratings)
    
if not insp.has_table("ratings_average_count", schema="db1"):
    df_ratings = pd.read_csv("ml-25m/ratings.csv")
    Insert_average_count_ratings(df_ratings[['movieId', 'rating']])
    
if not insp.has_table("tags", schema="db1"):
    df_tags = pd.read_csv("ml-25m/tags.csv")
    Insert_tags_table(df_tags)

if not insp.has_table("ratings_merged_small", schema="db1"):
    df_ratings_small = pd.read_csv('ml-100k/u.data', sep='\t',header=None,names=['userId','movieId','rating','timestamp'])
    # print(df_ratings.head())
    df_movies_small = pd.read_csv('ml-100k/u.item', sep='|',header=None,names='movieId | title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western'.split(' | ') ,encoding='latin-1')
    # print(items_dataset.head())
    ratings_merged = pd.merge(df_ratings_small, df_movies_small[['movieId','title']], how='inner', on='movieId')
    ratings_merged.index.name="ratingId"
    print(ratings_merged.head())
    Insert_ratings_merged_small_table(ratings_merged)

# print(df_movies.head())
# print(df_tags.head())

def get_movies_data():
    movies = session.query(Movies.movieId, Movies.title, Movies.genres)#.all()
    df_mov = orm_query(movies)
    return df_mov

def get_average_count_ratings():
    average_count_ratings = session.query(RatingsAverageCount.movieId, RatingsAverageCount.average, RatingsAverageCount.count)#.all()
    df_average_count_ratings = orm_query(average_count_ratings)
    return df_average_count_ratings

def get_ratings():
    print("Getting rating from database")
    ratings = session.query(Ratings.userId, Ratings.movieId, Ratings.rating, Ratings.timestamp)
    df_rat = orm_query(ratings)
    # if not df_rat.index.name == "movieId":
    #     df_rat.set_index(['movieId'], inplace=True)
    # df_rat.reset_index(inplace=True)
    print("Ratings are fetched")
    return df_rat
# my_ratings = session.query(Ratings).filter(Ratings.rating==5)#.first()

st.title("Welcome to Movies Database")

df_movies = get_movies_data()
df_movies_len = len(df_movies)
# taking page number as imput
st.write("## Movies List with Paginations")
page_size = st.number_input(label="Lines per page", min_value=1, max_value=20, value=8)
max_movies_page_num = int(df_movies_len/page_size)+1
print("max_movies_page_num:",max_movies_page_num)
page_num = st.number_input(label="page num:", min_value=1, max_value=max_movies_page_num, value=1)
st.write(paginate_dataframe(df_movies, int(page_size), int(page_num)))


st.write("## Search for a movie by title or part of the title.")
input_title_string = st.text_input(label="Enter title or part of the title", max_chars=80, value="Old Men")
st.write(df_movies.loc[df_movies['title'].str.contains(str(input_title_string), case=False)])

average_count_ratings = get_average_count_ratings()
st.write("## Search for the best rated films by genre")

input_genre_string = st.text_input(label="Input genre:", max_chars=80, value="Comedy")
df_movies_by_genre = pd.merge(df_movies.loc[df_movies['genres'].str.contains(str(input_genre_string), case=False)], average_count_ratings, on='movieId')
st.write(df_movies_by_genre.set_index(['title']).drop(['movieId',], axis=1).sort_values(["count", "average"], ascending = (False, False)))


st.write("## Bonus: Find users who have the same taste (users who rate the same movies similarly).")
rating_merged_df = get_ratings_merged_small()
model, userid_movieid_df, movies_knn_dataset = get_knn_dataset(rating_merged_df)
user = st.number_input(label="Enter user id:", min_value=1, max_value=200, value=8)
num_nearest_users = st.number_input(label="Enter number of similar users to find:", min_value=1, max_value=200, value=5)
users_nearest, dist = get_similar_users(model,userid_movieid_df,user, num_nearest_users)
st.write(pd.DataFrame({ 'Nearest Users': users_nearest, 'distance': dist}))


st.write("## Recommend the movie to the given user")
num_recommended_movies = st.number_input(label="Enter number movies to be recommended:", min_value=1, max_value=20, value=10)
movies_watched_by_given_users = movies_knn_dataset[movies_knn_dataset['userId'] == user]["movieId"].tolist()
movies_watched_by_similar_users = movies_knn_dataset[movies_knn_dataset['userId'].isin(users_nearest.tolist())]
movies_only_watched_by_similar_users_not_given_user = movies_watched_by_similar_users[~movies_watched_by_similar_users['movieId'].isin(movies_watched_by_given_users)]
# movies_watched_by_given_and_similar_users = movies_watched_by_similar_users[movies_watched_by_similar_users['movieId'].isin(movies_watched_by_given_users)]
# st.write("### Movies watched by both:", movies_watched_by_given_and_similar_users)
# print(movies_only_watched_by_similar_users_not_given_user[movies_only_watched_by_similar_users_not_given_user['movieId'].isin(movies_watched_by_given_users)])
movies_only_watched_by_similar_users_not_given_user = movies_only_watched_by_similar_users_not_given_user[['movieId', 'title', 'rating']].groupby(["movieId", 'title']).mean()
# movies_only_watched_by_similar_users_not_given_user.rename(columns={'rating': 'average'}, inplace=True)
# count_rated_movies_dict = df_ratings.groupby(["movieId"]).count()
# count_rated_movies_dict.rename(columns={'rating': 'count'}, inplace=True)
# average_count_ratings = pd.merge(mean_rated_movies_dict.loc[mean_rated_movies_dict["average"]>=4], count_rated_movies_dict.loc[count_rated_movies_dict["count"]>500], on='movieId')
st.dataframe(movies_only_watched_by_similar_users_not_given_user.sort_values(["rating"], ascending = (False)).head(num_recommended_movies))

# df_ratings = df_ratings = pd.read_csv("ml-25m/ratings.csv")
# get_knn_dataset_large(df_movies, df_ratings)

# mov_rtngs_sim_users = userid_movieid_df.values[users_nearest]
# # mov_rtngs_sim_users

# movies_list = userid_movieid_df.columns
# movies_list

# df_ratings = pd.read_csv('ml-100k/u.data', sep='\t',header=None,names=['userId','movieId','rating','timestamp'])
# print(df_ratings.head())
# df_movies = pd.read_csv('ml-100k/u.item', sep='|',header=None,names='movieId | title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western'.split(' | ') ,encoding='latin-1')
# print(df_movies.head())
# prepared_dataset = pd.merge(df_ratings, df_movies[['movieId','title']], how='inner', on='movieId')
# print(prepared_dataset.head())