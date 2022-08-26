# Basic libraries
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import unicodedata
import time
# from textblob import TextBlob
    

from wordcloud import WordCloud

# Natural Language Toolkit
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
from nltk.text import Text
# expanding contractions in Text Processing
import contractions

import tqdm

import nltk
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# gensim LDA model
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# TF-IDF kmeans model
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import mixture
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
# reduce vectors' dimensions
import umap.umap_ as umap
# conda install -c conda-forge umap-learn
from sklearn.manifold import TSNE


# import pyLDAvis.gensim
# conda install -c conda-forge pyldavis
import pyLDAvis.gensim_models

# ===========================================================================================        
# CHANGE 0 to 1 FOR EXECUTION
# Data understanding
data_understanding = 0 

# Data cleaning
gensim_lda_clean = 0
kmeans_clean = 1

# Building the models
# lda
gensim_lda_model = 0
gensim_lda_tuning = 0
# kmeans
kmeans_model = 1
kmeans_tuning = 0

# ===========================================================================================        
# Initializing variables
stop_words = stopwords.words('english')
stop_words_ext = stopwords.words('english')
stop_words_ext.extend(['thank', 'mr', 'new'])  
my_punctuations = '!"$%&\'()*+,-./”“:;<‘’’=>?[\\]^_`{|}~•@'
punctuations_sent = '"$%&\'()*+-/”“<‘’’=>[\\]^_`{|}~•@' # !,?.:; are removed
# Selection of optimal clusters 
# for lda, k = 9
lda_optimal_clusters = 9
# for kmeans, k = 6
kmeans_optimal_clusters = 6
# Representative words per topic
repr_words = 200
# set the pmi threshold
pmi_threshold = 5

# Regular expression patterns
retweets_pattern = r'(?<=RT\s)(@\s?[A-Za-z]+[A-Za-z0-9-_]+)'
mentioned_pattern = r'(?<!RT\s)(@\s?[A-Za-z]+[A-Za-z0-9-_]+)[^.]'
hashtags_pattern = r'(#\s?[A-Za-z]+[A-Za-z0-9-_]+)'
links1_pattern = r'(https?://\S+|www\.\S+)'
links2_pattern = r'(bit.ly/\S+)'
pic_pattern = r'pic.twitter.com/[a-zA-Z0-9]+'

# Stemming & Lemmatizing
lem = WordNetLemmatizer()
porter = PorterStemmer()

# Empty dictionaries
retweeted_dict = {}
mentioned_dict = {}
hashtags_dict = {}
url_dict = {}
pictures_dict = {}
word_freq = {}

# Empty lists
topics_list = []
# ===========================================================================================        
#                                   FUNCTIONS

# Preprocessing steps
def lemmatize_stemming(text):
    text_token_list = [porter.stem(lem.lemmatize(word, pos='v')) for word in text.split(' ')]
    return text_token_list


def remove_links(text):
    '''Taking a string and removing web links from it'''
    text = re.sub(links1_pattern, '', text)  # remove http links
    text = re.sub(links2_pattern, '', text)  # remove bitly links
    text = re.sub(pic_pattern, '', text)  # remove twitter pics
    return text


def remove_stopwords(text, stop_words):
    text_token_list = [word for word in text.split(' ') if word not in stop_words]
    return text_token_list


def remove_mentions(text):
    '''Taking a string and removing retweet and @user information'''
    text = re.sub(retweets_pattern, '', text) # remove retweet
    text = re.sub(mentioned_pattern, '', text) # remove tweeted at
    return text


def tweet_cleaner(text):
    global model_flag
    text = remove_mentions(text) # remove retweets
    text = remove_links(text) # remove links
    text = contractions.fix(text) # expand contractions  
    text = text.lower() # lowercase -- to avoid case sensitive issue
    
    # sentence_list = []
    # for word in text.split():
    #     if word in ['donald','trump','hilary', 'clinton', 'bernie', 'sanders', 'barack', 'obama']:
    #         sentence_list.append(word)

    #     else:
    #         sentence_list.append(str(TextBlob(word).correct()))
    # text = ' '.join(sentence_list)
    if model_flag == 0:
        text_token_list = remove_stopwords(text, stop_words) # remove stopwords
        text = ' '.join(text_token_list)
    # unicode normalization
    # (solution to both canonical and compatibility equivalence issues)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    text = re.sub('([0-9]+)', '', text) # numbers
    text = re.sub('\w*\d\w*', '', text) # numbers being part of a word (i.e., donaldtrump2020)
    
    if model_flag == 0:
        text = re.sub('['+ my_punctuations + ']+', ' ', text) # strip my_punctuations
    else:
        text = re.sub('['+ punctuations_sent + ']+', ' ', text) # strip punctuations_sent
        
    text = re.sub('\n', ' ', text) # newline tabulators
    text = re.sub('\s+', ' ', text) # double spacing
    
    text_token_list = lemmatize_stemming(text) # stemming and lemmatize
    text = ' '.join(text_token_list)
    if model_flag == 1:
        text_token_list = remove_stopwords(text, stop_words_ext) # remove stopwords among with specific ones that do not contribute
        text = ' '.join(text_token_list)
    return text


# Filter for ngrams with only noun-type structures
def ngram_filtering(ngram, ngram_flag):
    global my_stopwords
    tag = nltk.pos_tag(ngram)
    if tag[0][1] not in ['JJ', 'NN'] and tag[1][1] not in ['JJ','NN']:
        return False
    if ngram_flag == 2:
        if ngram[0] in my_stopwords or ngram[1] in my_stopwords:
            return False
    if ngram_flag == 3:
        if ngram[0] in my_stopwords or ngram[-1] in my_stopwords or ngram[1] in my_stopwords:
            return False
    if 'n' in ngram or 't' in ngram:
         return False
    if 'PRON' in ngram:
        return False
    return True 

# Concatenate n-grams
def replace_ngram(x):
    for gram in trigrams:
        x = x.replace(gram, '_'.join(gram.split()))
    for gram in bigrams:
        x = x.replace(gram, '_'.join(gram.split()))
    return x


# Filter for only nouns
def noun_only(x):
    pos_comment = nltk.pos_tag(x)
    filtered = [word[0] for word in pos_comment if word[1] in ['NN']]
    # to filter both noun and verbs
    #filtered = [word[0] for word in pos_comment if word[1] in ['NN','VB', 'VBD', 'VBG', 'VBN', 'VBZ']]
    return filtered


# Gensim LDA model

# Computing coherence
def compute_coherence_values(corpus, dictionary, text, k, a, b, final_model):
    lda_model = gensim.models.LdaMulticore(corpus = corpus,
                                           id2word = dictionary,
                                           num_topics = k, 
                                           random_state = 100,
                                           chunksize = 100,
                                           passes = 400,
                                           alpha = a,
                                           eta = b)
    coherence_model_lda = CoherenceModel(model = lda_model, texts = text, 
                                         dictionary = dictionary, coherence = 'c_v')
    
    if final_model == 1:
        
        # Compute Coherence Score
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        
        # Compute Perplexity
        perplexity_lda = lda_model.log_perplexity(corpus)
        print('\nPerplexity: ', perplexity_lda) 
        # Perplexity:  -8.253994961551092 for k = 9

        return lda_model
    
    return coherence_model_lda.get_coherence()


# TF-IDF kmeans model

# get tfidf of documents
def get_tfidf_embedding(items):
  # tfidf = TfidfVectorizer()
  tfidf = TfidfVectorizer(stop_words = 'english')
  embeddings = tfidf.fit_transform(items)
  tfidf_feature_names = tfidf.get_feature_names()
  return embeddings, tfidf_feature_names

# find optimal number of clusters
def find_optimal_clusters(x, method):
    maxClusters = 15
    K = range(2, maxClusters)
    if method == "elbow":
        SSE = []
        for k in K:
            kmeans = MiniBatchKMeans(n_clusters = k, batch_size = 300)
            kmeans.fit(x)
            SSE.append(kmeans.inertia_)      
           
        plt.plot(K, SSE,'bx-')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia Values')
        plt.show()
        
    if method == "BIC":
        # starting time of computation
        start = time.time()
        # pca = PCA(n_components=20)
        # embedding_tf_idf = pca.fit(embedding_tf_idf.todense()).transform(embedding_tf_idf.todense())
        bicAll = []
        logLikelihood = []
        clustersAll = []
        silhouettesAll = []

        for n in K:
            gmm = mixture.GaussianMixture(n_components = n, covariance_type = 'full').fit(x)
            logLikelihood.append(gmm.score(x))
            clustersAll.append(n)
            bicAll.append(gmm.bic(x))
            labels = gmm.predict(x)
            silhouetteScore = metrics.silhouette_score(x, labels)
            silhouettesAll.append(silhouetteScore)
        # print 'mixtures, clusters', n, gmm.bic(x)
        
        # ending time of computation
        end = time.time()
        # duration for finding the optimal number clusters after dimensions reduction using umap
        print("Execution time for finding optimal number of clusters", end-start)
        
        plt.figure(4)
        plt.title('Gaussian Mixture with umap reduce')
        plt.xlabel('Number of Clusters')
        plt.ylabel('BIC Values')
        plt.plot(clustersAll, bicAll, '*-')
        plt.show()
        return bicAll, logLikelihood, clustersAll, silhouettesAll

# plot topics in clusters
def plot_embeddings(embedding, labels, title):
    labels = np.array(labels)
    distinct_labels = set(labels)
    
    n = len(embedding)
    counter = collections.Counter(labels)
    for i in range(len(distinct_labels)):
        ratio = (counter[i] / n ) * 100
        cluster_label = f"cluster {i}: { round(ratio,2)}"
        x = embedding[:, 0][labels == i]
        y = embedding[:, 1][labels == i]
        plt.plot(x, y, '.', alpha = 0.4, label = cluster_label)
    plt.legend(title = "Topic", loc = 'upper left', bbox_to_anchor = (1.01, 1))
    plt.title(title)
    

# reduce the dimensions of the embeddings using:
# UMAP method
def reduce_umap(embedding, umap_plot):
    if umap_plot == 1:
        reducer = umap.UMAP()
    if umap_plot == 0:
          reducer = umap.UMAP(n_components = 10) 
    embedding_umap = reducer.fit_transform(embedding)
    return embedding_umap

# TSNE method
def reduce_tsne(embedding):
    tsne = TSNE(n_components = 2)
    reduced = tsne.fit_transform(embedding)
    return reduced


# apply kmeans to embeddings
def kmeans_fitting(embeddings, num_topics):
    kmeans_model = KMeans(num_topics)
    kmeans_model.fit(embeddings)
    topics_labels = kmeans_model.predict(embeddings)
    return topics_labels

# predict topics with kmeans
def predict_topics_with_kmeans(embeddings, feature_names, num_topics, miniBatchKMeans):
    global topics_list, repr_words
    # with MiniBatchKMeans
    if miniBatchKMeans == 1:
        kmeans_model = MiniBatchKMeans(n_clusters = num_topics)
    # without MiniBatchKMeans
    if miniBatchKMeans == 0:
        kmeans_model = KMeans(num_topics)
        
    kmeans_model.fit(embeddings)
    centers = kmeans_model.cluster_centers_.argsort()[:,::-1]
    
    # print the representative words of each topic
    for i in range(0, num_topics):
        word_list = []
        print("Cluster%d:"% i)
        # number of representative words in topics
        for j in centers[i, :repr_words]:
            word_list.append(feature_names[j])
        print(word_list)
        topics_list.append(word_list)
    return topics_list


# LOADING THE DATA
# Change the path of the directory accordingly  
path = 'D:/Data Science/SPRING TERM/NATURAL LANGUAGE PROCESSING/Group Project/Datasets/'       
# Open csv file
trump = pd.read_csv(path + 'trumptweets.csv', lineterminator = '\n', parse_dates = True)

# ===========================================================================================        
#                                           DATA UNDERSTANDING 

if data_understanding == 1:
    print('A preview of the data...')
    # Trump dataset
    print(trump.head())
    # to get a preview of the data

    print('\nThe tweet collection info...')
    # Trump dataset
    print(trump.columns)

    # The dimensions of the datasets
    print('\nTrump dataset shape: {}'.format(trump.shape))

    # General information
    print('\nTrump dataset information')
    print(trump.info(memory_usage='deep'))

    print("\nDescriptive Statistics")
    # Trump dataset
    print(trump.describe())

print('\nColumns of the dataset: ', list(trump.columns))

print('\nNumber of tweets: ', len(trump.content))


# ===========================================================================================        
#                                           DATA EXPLORATION 

for contents in trump.content:
    retweets = re.findall(retweets_pattern, contents)
    mentioned = re.findall(mentioned_pattern, remove_links(contents))
    hashtags = re.findall(hashtags_pattern, remove_links(contents))
    links1 = re.findall(links1_pattern, contents)
    links2 = re.findall(links2_pattern, contents)
    pic_url = re.findall(pic_pattern, contents)            
            
    if len(retweets) > 0:
        if retweets[0] not in retweeted_dict:
            retweeted_dict[retweets[0]] = 0
        retweeted_dict[retweets[0]] += 1
    if len(mentioned) > 0:
        for mention in mentioned:
            if mention not in mentioned_dict:
                mentioned_dict[mention] = 0
            mentioned_dict[mention] += 1
    if len(hashtags) > 0:
        for hashtag in hashtags:
            if hashtag not in hashtags_dict:
                hashtags_dict[hashtag] = 0
            hashtags_dict[hashtag] += 1
    if len(links1) > 0:
        for link in links1:
            if link not in url_dict:
                url_dict[link] = 0
            url_dict[link] += 1
    if len(links2) > 0:
        for link in links2:
            if link not in url_dict:
                url_dict[link] = 0
            url_dict[link] += 1
    if len(pic_url)>0:
        for pic in pic_url:
            if pic not in pictures_dict:
                pictures_dict[pic] = 0
            pictures_dict[pic] += 1
            
print('\nNumber of retweets: ', sum(retweeted_dict.values()))
print('\nNumber of mentioned: ', sum(mentioned_dict.values()))
print('\nNumber of hashtags: ', sum(hashtags_dict.values()))
print('\nNumber of links: ', sum(url_dict.values()))
print('\nNumber of picture urls: ', sum(pictures_dict.values()))

# Sorting the dictionaries by value in descending order
retweeted_dict = dict(sorted(retweeted_dict.items(), key=lambda x: (-x[1], x[0])))
mentioned_dict = dict(sorted(mentioned_dict.items(), key=lambda x: (-x[1], x[0])))
hashtags_dict = dict(sorted(hashtags_dict.items(), key=lambda x: (-x[1], x[0])))
url_dict = dict(sorted(url_dict.items(), key=lambda x: (-x[1], x[0])))
pictures_dict = dict(sorted(pictures_dict.items(), key=lambda x: (-x[1], x[0])))

# Saving most common to html format
retweeted_df = pd.DataFrame(retweeted_dict.items(), columns=['Retweets', 'Frequency'])
fig = px.bar(retweeted_df[:20], x='Retweets', y='Frequency')
fig.write_html(path +"retweets.html")
fig.show()

mentioned_df = pd.DataFrame(mentioned_dict.items(), columns=['Mentioned', 'Frequency'])
fig = px.bar(mentioned_df[:20], x='Mentioned', y='Frequency')
fig.write_html(path +"mentions.html")
fig.show()

hashtag_df = pd.DataFrame(hashtags_dict.items(), columns=['Hashtag', 'Frequency'])
fig = px.bar(hashtag_df[:20], x='Hashtag', y='Frequency')
fig.write_html(path +"hashtag.html")
fig.show()

url_df = pd.DataFrame(url_dict.items(), columns=['URL', 'Frequency'])
fig = px.bar(url_df[:20], x='URL', y='Frequency')
fig.write_html(path +"url.html")
fig.show()

pic_df = pd.DataFrame(pictures_dict.items(), columns=['Tweet picture', 'Frequency'])
fig = px.bar(pic_df[:20], x='Tweet picture', y='Frequency')
fig.write_html(path +"pic.html")
fig.show()


# ===========================================================================================        
#                                       DATA CLEANING

# For gensim model
if gensim_lda_clean ==  1:
    # 0 / 1 : 0 for gensim / bert lda model, respectively
    model_flag = 0
    # Cleaning tweets
    trump['gensim_content'] = trump['content'].apply(lambda x: tweet_cleaner(x))
    
    def cleaner(text):
        text_token_list = remove_stopwords(text, my_stopwords) # remove stopwords
        text = ' '.join(text_token_list)
        return text
        
    trump['gensim_content'] = trump['gensim_content'].apply(lambda x: cleaner(x))
    
    # Checking for NaN values 
    if trump['gensim_content'].isnull().values.any() == False:
        print('There are no NaN values after the preprocessing')   
    
    # most common words/tokens after stopword and punctuation removal
    for tweet in trump.gensim_content:
        word_list = word_tokenize(tweet)
        for word in word_list:
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1          
    
    print('\nNumber of tokens: ', sum(word_freq.values()))
    
    # Sorting dictionary by value in descending order
    word_freq = dict(sorted(word_freq.items(), key=lambda x: (-x[1], x[0])))
    
    # Saving most common to html format
    word_freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])
    fig = px.bar(word_freq_df[:20], x='Word', y='Frequency')
    fig.write_html(path +"word.html")
    fig.show()
    
    # collocations (phrases/expressions that are likely to co-occur)
    trump_collocations = Text(tweet for tweet in trump['gensim_content']).collocation_list(num=20)
    print(f'The 20 most common collocations are: {trump_collocations}')
    

    # Steps to Optimize Interpretability
    
    # STEP 1
    # Identify phrases through n-grams 
    # Using Pointwise Mutual Information score to identify significant bigrams and trigrams
    # 2-grams
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = nltk.collocations.BigramCollocationFinder.from_documents([comment.split() for comment in trump.gensim_content])
    # Filter only those that occur at least 50 times
    finder.apply_freq_filter(50)
    bigram_scores = finder.score_ngrams(bigram_measures.pmi)

    bigram_pmi = pd.DataFrame(bigram_scores)
    bigram_pmi.columns = ['bigram', 'pmi']
    bigram_pmi.sort_values(by = 'pmi', axis = 0, ascending = False, inplace = True)
    print(bigram_pmi)
    
    # 3-grams
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder = nltk.collocations.TrigramCollocationFinder.from_documents([comment.split() for comment in trump.gensim_content])
    # Filter only those that occur at least 50 times
    finder.apply_freq_filter(50)
    trigram_scores = finder.score_ngrams(trigram_measures.pmi)    

    trigram_pmi = pd.DataFrame(trigram_scores)
    trigram_pmi.columns = ['trigram', 'pmi']
    trigram_pmi.sort_values(by = 'pmi', axis = 0, ascending = False, inplace = True)
    print(trigram_pmi) 
    
    # filter noun-type structures
    
    # set pmi threshold (where n-grams stop making sense)
    # choose top 500 ngrams in this case ranked by PMI that have noun like structures
    filtered_bigram = bigram_pmi[bigram_pmi.apply(lambda bigram: ngram_filtering(bigram['bigram'], ngram_flag = 2) 
                                                  and bigram.pmi > pmi_threshold, axis = 1)][:500]
    
    filtered_trigram = trigram_pmi[trigram_pmi.apply(lambda trigram: ngram_filtering(trigram['trigram'], ngram_flag = 3)
                                                     and trigram.pmi > pmi_threshold, axis = 1)][:500]
    
    
    bigrams = [' '.join(x) for x in filtered_bigram.bigram.values if len(x[0]) > 2 or len(x[1]) > 2]
    trigrams = [' '.join(x) for x in filtered_trigram.trigram.values if len(x[0]) > 2 or len(x[1]) > 2 and len(x[2]) > 2]

    # Sampling of bigrams
    sample_bigrams = bigrams[:10]
    print("A sample of bigrams")
    print(sample_bigrams)
    
    # Sampling of trigrams
    sample_trigrams = trigrams[:10]
    print("A sample of trigrams")
    print(sample_trigrams)
    
    # concatenate n-grams
    # keeping only the column: gensim content after tweet cleaning
    tweets_w_ngrams = trump['gensim_content'].copy()
    tweets_w_ngrams = tweets_w_ngrams.map(lambda x: replace_ngram(x))
    # tokenize tweets, remove stop words and words with less than 2 characters
    tweets_w_ngrams = tweets_w_ngrams.map(lambda x: [word for word in x.split()
                                                    if word not in my_stopwords 
                                                    and len(word) > 2]) 
    print(tweets_w_ngrams.head())
    
    
    # Filter for only nouns
    # Nouns are most likely indicators of a topic, therefore filtering for the 
    # noun cleans the text for words that are more interpretable in the topic model
    final_tweets = tweets_w_ngrams.map(noun_only)    
    

# For kmeans model
if kmeans_clean == 1:
    # 0 / 1 : 0 for gensim / bert lda model, respectively
    model_flag = 1
    # Cleaning tweets
    trump['tfidf_content'] = trump['content'].apply(lambda x: tweet_cleaner(x))
    tfidf_sentences = trump['tfidf_content'].tolist() 

# =========================================================================================== 
#                               BUILDING LDA MODEL

# For gensim LDA
if gensim_lda_model == 1:
    # trump_sample = trump['gensim_content'].sample(n = 10000, random_state = 1)  
    trump_sample = final_tweets
    
    # Create a dictionary from trump_sample containing the 
    # number of times a word appears in the training set
    dictionary = corpora.Dictionary(trump_sample)
    # dictionary.filter_extremes(no_below = 15, no_above = 0.5, keep_n = 100000)
    
    # Term Document Frequency
    corpus = [dictionary.doc2bow(doc) for doc in trump_sample]
    
    
    #                       Hyperparameter Tuning
    if gensim_lda_tuning == 1:
    
        grid = {}
        grid['Validation_Set'] = {}
        
        # Topics range
        min_topics = 5
        max_topics = 9
        step_size = 1
        topics_range = range(min_topics, max_topics, step_size)
        
        # Alpha parameter
        alpha = list(np.arange(0.01, 1, 0.3))
        alpha.append('symmetric')
        alpha.append('asymmetric')
        
        # Beta parameter
        beta = list(np.arange(0.01, 1, 0.3))
        beta.append('symmetric')
        
        # Validation sets
        num_of_docs = len(corpus)
        
        corpus_title = ['100% Corpus']
        model_results = {'Validation_Set': [],
                         'Topics': [],
                         'Alpha': [],
                         'Beta': [],
                         'Coherence': []
                        }
        
        pbar = tqdm.tqdm(total = 540)
        
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv = compute_coherence_values(corpus = corpus, dictionary = dictionary, text = trump_sample,
                                                  k = k, a = a, b = b, final_model = 0)
                    # Save the model results
                    model_results['Validation_Set'].append(corpus)
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)
                    
                    pbar.update(1)
        hyperparameters = pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index = False)
        pbar.close()
        
        hyperparameters = pd.read_csv('lda_tuning_results.csv')
        # Sorting by coherence
        hyperparameters = hyperparameters.sort_values(by = ['Coherence'], ascending = False)
        hyperparameters.plot.scatter(x = 'Topics', y = 'Coherence')
        

        
    # Build LDA model

    # Final Model
    # starting time of computation
    start = time.time()
    lda_model = compute_coherence_values(corpus, dictionary, text = trump_sample, k = 9, a = 'asymmetric', b = 0.61, final_model = 1)
    # ending time of computation
    end = time.time()
    # duration for LDA model
    print("Execution time for LDA model", end-start)
    
    # Coherence Score:  0.35928287423947913 for 8
    # Coherence Score:  0.3513624278564942 for 6
    # Coherence Score:  0.36802913236695517 for 9
    
    # Coherence Score:  0.349881229874554 (for 9 TextBlob)
    # Perplexity:  -8.089329257696422 (for 9 TextBlob)
    
    
    # Topic keywords and their weightage (importance) 
    types = lda_model.show_topics()
    for t in types:
        print(t)
        print('----------------')
     
    # Topics in dataframe
    topics = {}
    for i in range(lda_optimal_clusters):
        words = lda_model.show_topic(i, topn = 20)
        # print(words)
        topics["Topic number" + "{}".format(i)] = [i[0] for i in words]
        
    df_topics = pd.DataFrame(topics)
    
    # Topic visualization
    Vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics = False)
    pyLDAvis.display(Vis)
    
    # Exporting pyLDAvis graphs as standalone webpage
    pyLDAvis.save_html(Vis, path + 'lda.html')
    
    
    
    # Word Cloud of Topics
    for topic in range(lda_model.num_topics):
        plt.figure()
        plt.imshow(WordCloud().fit_words(dict(lda_model.show_topic(topic, 200))))
        plt.axis("off")
        plt.title("Topic #" + str(topic))
        plt.show()

# ========================================================================================    
#                               BUILDING k-means MODEL
if kmeans_model == 1:
    # Generate embedding with tfidf
    embedding_tf_idf, tfidf_feature_names = get_tfidf_embedding(tfidf_sentences)
    print("Shape of sentences applied tf-idf :", embedding_tf_idf.shape)
    print("Type of tf-idf vector :", type(embedding_tf_idf[0]))
    print("Sample of tf-idf vector :", embedding_tf_idf[0])  

    if kmeans_tuning == 1:
        # Finding Optimal Clusters
        # Plot the SSE for a range of cluster sizes = 14 
        elbow_method = find_optimal_clusters(embedding_tf_idf, method = "elbow")
        # Plot the BIC 
        bic_method = find_optimal_clusters(reduce_umap(embedding_tf_idf, umap_plot = 0), method = "BIC")
        print(bic_method)
        # We look for the "elbow" where the SSE begins to level off.
        
        
        # Plotting the Topics in Clusters
        # Apply kmeans to raw vectors
        labels_tfidf_raw = kmeans_fitting(embedding_tf_idf, num_topics = kmeans_optimal_clusters)
        print("Embedding Tf-idf shape :", embedding_tf_idf.shape)  
        # Embedding Tf-idf shape : (41122, 21173)
        
        # Apply kmeans to umap vectors - interpreting both distances between positions of points and clusters
        embedding_tf_idf_umap = reduce_umap(embedding_tf_idf, umap_plot = 1)
        labels_tfidf_umap  = kmeans_fitting(embedding_tf_idf_umap, num_topics = kmeans_optimal_clusters)
        print("Embedding shape after umap", embedding_tf_idf_umap.shape)
        # Embedding shape after umap (41122, 2)
        plot_embeddings(embedding_tf_idf_umap, labels_tfidf_umap, "Tf-idf with Umap")
    
    
        # Apply kmeans to tsne vectors - better at capturing relations between neighbors
        embedding_tf_idf_tsne = reduce_tsne(embedding_tf_idf)
        labels_tfidf_tsne = kmeans_fitting(embedding_tf_idf_tsne, num_topics = kmeans_optimal_clusters)
        
        plot_embeddings(embedding_tf_idf_tsne, labels_tfidf_tsne, "Tf-idf with T-sne")
        
        # Silhouette score
        print("Silhouette score:" )
        print("without dimension's reduction :", silhouette_score(embedding_tf_idf , labels_tfidf_raw))
        # without dimension's reduction : 0.010978946753918436
        print("with Tf-idf   T-sne   :", silhouette_score(embedding_tf_idf_tsne, labels_tfidf_tsne))
        # with Tf-idf   T-sne   : 0.32096392
        print("with Tf-idf   Umap    :", silhouette_score(embedding_tf_idf_umap, labels_tfidf_umap))
        # with Tf-idf   Umap    : 0.4062511
    


    # Predicting the Topics of Tweets
    # Topics with MiniBatchKMeans
    MiniBatchkmeansTopics = predict_topics_with_kmeans(embedding_tf_idf, feature_names = tfidf_feature_names, 
                                                       num_topics = kmeans_optimal_clusters, miniBatchKMeans = 1)
    # Reinitializing topics_list 
    topics_list = []
    
    # Topics without MiniBatchKMeans
    
    # starting time of computation
    start = time.time()
    kmeansTopics = predict_topics_with_kmeans(embedding_tf_idf, feature_names = tfidf_feature_names, 
                                              num_topics = kmeans_optimal_clusters, miniBatchKMeans = 0)
    
    # ending time of computation
    end = time.time()
    # duration for LDA model
    print("Execution time for LDA model", end-start)    
    
    
    # Visualization of Topics - Word Cloud 
    for topics in range(kmeans_optimal_clusters):
        # convert list to string and generate
        unique_string = (" ").join(kmeansTopics[topics])
        wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
        plt.figure(figsize = (15,8))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.title("Topic #" + str(topics))
        plt.show()
        plt.savefig("kmeans topic modelling" + str(topics) + ".png", bbox_inches = 'tight')
        # plt.show()


