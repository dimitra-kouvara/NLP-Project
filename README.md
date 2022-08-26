# NLP Project: Topic Modeling on Trump Tweets
This project is an attempt to give structure to this highly unstructured data (dataset: Trump’s tweets until June 2020),  detect word patterns in those data and cluster those words in groups that best characterize  the different topics that the former president tweets about. The method used is topic modeling and provides us with ways to organize, understand and summarize large collections of textual information. It helps us discover hidden topical patterns that are present across the collection, annotate documents according to these topics and organize, search, and summarize texts.


INSTRUCTIONS ON HOW TO RUN NLP_project.py

Follow the link: https://www.kaggle.com/austinreese/trump-tweets
Download trumptweets.csv file and save to the same directory as NLP_project.py

1. import the necessary libraries
2. the code is set to run in whole
   In case you want to execute LDA model and kmeans seperately,
   change 0 and 1 accordingly (lines 56-70)
   For LDA model
   gensim_lda_clean = 1
   
   gensim_lda_model = 1
   
   gensim_lda_tuning = 0

   Note: it is suggested not to activate gensim_lda_tuning as its 
   execution time is rather long.

   For kmeans
   kmeans_clean = 0
   
   kmeans_model = 0
   
   kmeans_tuning = 0
