# Simple Content Based Recommendation System
# - Using genres to weigh likeliness of being recommended using graph plotting (x,y)
# - Cosine Similarity A*B / (||A|| *||B||)
# - For more complex recommendations, matrix increases, but same process is used.

import pandas

# Create new column labels
ratings_features = ["user_id", "item_id", "rating", "timestamp"]

# Create dataframe using new labels
ratings_dataframe = pandas.read_csv("ratings.csv", names = ratings_features)

# Call dataframe (if using in Visual Studio, will need pandas, Jupyter, miniconda [latest python])
# if using Google Colab, simply upload the 2 files (ratings.csv and movies.csv - attached to git repository)
ratings_dataframe

# Variable used for dropping the defined row(s)
FIRST_INDEX_ROW = 0

# Setting the dataframe format
ratings_dataframe = ratings_dataframe.drop(ratings_dataframe.index[FIRST_INDEX_ROW])

# Call dataframe
ratings_dataframe

# Display dataframe content types
ratings_dataframe.info()

# Convert dataframe data types to float
ratings_dataframe = ratings_dataframe.astype("float")

# Display dataframe content types
ratings_dataframe.info()

# Display formated dataframe
ratings_dataframe

# Define new dataframe
movies_dataframe = pandas.read_csv("movies.csv")

# Call dataframe
movies_dataframe

# Display dataframe content types
movies_dataframe.info()

# Reform dataframe based on movie titles
movie_titles_dataframe = movies_dataframe[["movieId","title"]]

# Display formated dataframe
movie_titles_dataframe

# Display dataframe content types
movie_titles_dataframe.info()

# Cast movie_titles_dataframe column "movieID" to float after converting to string (str)
movie_titles_dataframe["movieId"] = movie_titles_dataframe["movieId"].astype(str).astype(float)

# Display dataframe content types
movie_titles_dataframe.info()

# Merge the two dataframes (ratings_dataframe & movie_titles_dataframe) on the "movieId" column
merged_dataframe = pandas.merge(ratings_dataframe, movie_titles_dataframe, on = "movieId")

# Display merged dataframe content
merged_dataframe

# Group results by "movieId", and count results of "rating", sorted descending
merged_dataframe.groupby("movieId")["rating"].count().sort_values(ascending=False)

# Create a pivot table variable using the merged dataframe
crosstab = merged_dataframe.pivot_table(values = "rating",
                              index = "userId",
                             columns = "title",
                             fill_value=0)
# Display pivot table (crosstab) content
crosstab

# Assign "X" as the transposed pivot table (columns become rows)
X = crosstab.T

# Display transposed (crosstab) content
X

# Importing decomposition from TruncatedSVD library
from sklearn.decomposition import TruncatedSVD

# Defining decomposition, and type (ex. matrix)
NUMBER_OF_COMPONENTS = 12
singular_value_decomposition = TruncatedSVD(n_components= NUMBER_OF_COMPONENTS,
                                            random_state=1)
matrix = singular_value_decomposition.fit_transform(X)

# Displaying matrix content
matrix

# Importing numpy library
import numpy

# Calling numby's "corrcoef" (correlation coefficent)
correlation_matrix = numpy.corrcoef(matrix)

# Display content
correlation_matrix

# Define movie list using the crosstab variable's columns
movie_titles = crosstab.columns

# Display content
movie_titles

# Convert the movie_title dataframe to a list
movies_list = list(movie_titles)

# Display list content
movies_list

# Search list for specific movie to correlate to
example_movie_index = movies_list.index("Batman: Year One (2011)")

# Display content
example_movie_index

# Building the correlation data from the correlation_matrix using the example_movie_index as the initial criteria, then displaying result
example_correlation = correlation_matrix[example_movie_index]
example_correlation

# Defining filter criteria MIN and MAX correlation values
MAXIMUM_CORRELATION = 1.0
MINIMUM_CORRELATION = 0.85

# Display list of movies matching correlation values (i.e "Which other movies are likely to be enjoyed based on the example_movie_index selected")
list(movie_titles[(example_correlation < MAXIMUM_CORRELATION)& (example_correlation > MINIMUM_CORRELATION)])
