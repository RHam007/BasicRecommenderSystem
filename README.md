The .py is intended for use within Google's Colab, but can be used in other IDE's that support Jupyter (or Colab) intergration to display the defined dataframes.
If unsure how to do this, please do a websearch for how to use colab with your IDE and/or how to display colab dataframes in your IDE.

The original content was created as part of the Mammoth Interactive class "Creating Your First Movie Recommender System: A Comprehensive Guide to Building Basic Film Suggestion Engines"
Comments were added by me, as I was working through the materials, to better clarify what was happening at each step.  Hopefully these comments help you as well!

Data Sources and References below:

"""Basic Movie Recommender System.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hG7xi7duSuN6hVrTidOvfAqv02DP6GVs

https://colab.research.google.com/

Data Download Link: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

Data Source: https://grouplens.org/datasets/movielens/latest/
"""

'''
@article{10.1145/2827872,
author = {Harper, F. Maxwell and Konstan, Joseph A.},
title = {The MovieLens Datasets: History and Context},
year = {2015},
issue_date = {January 2016},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {5},
number = {4},
issn = {2160-6455},
url = {https://doi.org/10.1145/2827872},
doi = {10.1145/2827872},
journal = {ACM Trans. Interact. Intell. Syst.},
month = dec,
articleno = {19},
numpages = {19},
keywords = {Datasets, recommendations, ratings, MovieLens}
}
'''
