# Word processing
from wordcloud import WordCloud, STOPWORDS
from sklearn import preprocessing
from gensim import models, corpora

# data processing
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# nltk import
from nltk.tag import PerceptronTagger
from nltk.corpus import stopwords, subjectivity
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk.data

# utils
import pycountry
import random
from PIL import Image
from os import path
from os.path import exists
from collections import Counter 


# Set of english token
SENTENCES_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')

# tools to detect sentiments in words
SID = SentimentIntensityAnalyzer()

# Regex Tokenizer 
REGEX_TOKENIZER = RegexpTokenizer(r'\w+')

def generate_raw_text(data):
    '''
    Generate a String from an array of string
    attributes:
        - data : array of String to transform
    
    return value String
    '''
    text = ''
    for d in data:
        text += str(d) + ' '
    return text


def do_stemming_words(stemmer, words):
    '''
    Generate a String applying a stemmer on each words of an array
    attributes:
        - stemmer : apply stemmer to each word
        - words   : array of String to transform
    
    return value String
    '''
    text = ''
    for w in words:
        text += stemmer.stem(w)
    return text


def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(40, 60)


def generate_word_cloud(text, img_name='envelope.png', max_words=1000, width=900, height=900, dpi=400, file_name=None):
    '''
    Generate and display a word cloud from a text removing tokens (STOPWORD)
    attributes:
        - text      : text to study
        - img_name  : name of the picture used for the word cloud.
        - max_words : limit max of words
        - width     : width of the picture
        - height    : height of the picture
        - dpi       : quality of the picture
        - file_name : name of the picture in case you want to save the picture (the picture is saved in ./images/) 
    '''
    stopwords = set(STOPWORDS)
    mask = np.array(Image.open(img_name))
    wc = WordCloud(background_color="white", mask=mask, max_words=max_words, stopwords=stopwords).generate(text)
    plt.figure(figsize=(9, 9), dpi=dpi)
    plt.axis("off")
    plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3))
    if file_name:
        path = './images/' + file_name + '.png'
        if not exists(path):
            plt.savefig(path, dpi=dpi)
    return plt


def get_country_name(word):
    '''
    Get the contry if exist in dictionnary
    attributes:
        - word : text to study
        
    return value None or the name of the contry if exist in dictionnary
    '''
    lower_word = str.lower(word)
    for c in pycountry.countries:
        if (word == c.alpha_2) or (word == c.alpha_3) or (lower_word == str.lower(c.name)) or (hasattr(c, 'official_name') and (lower_word == str.lower(c.official_name))):
            return c.name
    return None


def check_if_country_in_text(country, text):
    '''
    return the number of time, the text contain the contry name
    
    attributes:
        - country :
        - text   : text to study
        
    return value : boolean
    '''
    return ((str.lower(country.name) in text) or (hasattr(country, 'official_name') and (str.lower(country.official_name) in text)))

def count_country_occurrences(country, text):
    '''
    return the number of time, the text contain the contry name
    attributes:
        - country : name of the contry
        - text    : text to study
        
    return value : integer
    ''' 
    nb_occurrences = 0
    #nb_occurrences += text.count(country.alpha_2)
    #nb_occurrences += text.count(country.alpha_3)
    nb_occurrences += text.count(str.lower(country.name))
    if hasattr(country, 'official_name'):
        nb_occurrences += text.count(str.lower(country.official_name))
    return nb_occurrences


def count_countries_occurrences(text):
    '''
    Count for each country the number of time the country appear in the text
    attributes:
        - text : text to study
        
    return value : array containing for each contry the number of apparition.
    '''
    lower_text = str.lower(text)
    countries = Counter()
    for country in pycountry.countries:
        nb_occurrences = count_country_occurrences(country, lower_text)
        countries[country.name] = nb_occurrences
    return countries

    
def get_wordnet_tag_type(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def get_sentiwordnet_scores(tokens):   
    '''
    Associate a feeling POSITIVE / NEGATIVE from words present in token
    attributes:
        - tokens : text to study
        
    return value : map containing the number of positive/negative values
    '''
    types = Counter({'Positive': 0, 'Negative': 0})
    for word, pos_tag in nltk.pos_tag(tokens):
        tag = get_wordnet_tag_type(pos_tag)
        synset_list = list(swn.senti_synsets(word, pos=tag))
        if synset_list:
            types['Positive'] += synset_list[0].pos_score()
            types['Negative'] += synset_list[0].neg_score()
        else:
            continue
    return types


def get_vader_scores(email_content):
    '''
    Associate a feeling POSITIVE / NEGATIVE from words present in email_content
    attributes:
        - email_content : text to study
        
    return value : map containing the number of positive/negative values
    '''
    types = Counter({'Positive': 0, 'Negative': 0})
    tokens = SENTENCES_DETECTOR.tokenize(email_content.strip())
    for sentence in tokens:
        scores = SID.polarity_scores(sentence)
        types['Positive'] += scores['pos']
        types['Negative'] += scores['neg']
    return types
 
def retrieve_email_sentiment(email, analyzer='sentiwordnet'):
  
    email_content = str(email['ExtractedSubject']) + ' ' + str(email['ExtractedBodyText'])
    if(analyzer == 'Vader'):
        types = get_vader_scores(email_content)
    else:
        tokens = REGEX_TOKENIZER.tokenize(email_content)
        types = get_sentiwordnet_scores(tokens)
    
    if types['Positive'] > abs(types['Negative']):
        email['Type'] = 'Positive'
    elif abs(types['Negative']) > types['Positive']:
        email['Type'] = 'Negative'
    else:
        email['Type'] = 'Neutral'
        
    return email


def get_countries_sentiment(emails):
    countries_sentiment = Counter()
    for index, email in emails.iterrows():
        email_content = str(email['ExtractedSubject']) + ' ' + str(email['ExtractedBodyText'])
        lower_email_content = str.lower(email_content)
        for country in pycountry.countries:
            is_in_text = check_if_country_in_text(country, lower_email_content)
            if(is_in_text == True):
                if email['Type'] == 'Positive':
                    countries_sentiment[country.name] += 1
                elif email['Type'] == 'Negative':
                    countries_sentiment[country.name] -= 1
                else:
                    countries_sentiment[country.name] += 0
    return countries_sentiment


def plot_sentiment_by_contry(data_mails,opt,nb_contry = 20):
    '''
    Plot the number of country in axis, the number of occurence in ordinate
    and use 4 colors for the sentiments associate with the country. 
    attributes: 
        - opt : allow to select the option of the graph: 
                - 'only_good' = keep only good feeling about the contry
                - 'only_bad' = keep only bad feeling about the contry
                -            = concat the good and bad feeling in the same plot.
        - data_mails : dataframe containing the data of the mails
        - nb_contry : Selection on the most representative country
    '''
    # Selecting interesting data for the plot.
    if opt == 'only_good':
        data_plot = data_mails.nlargest(nb_contry, 'Sentiment')
        title = 'Hilary\'s opinion on the ' + str(nb_contry) + ' best feelings about contry'
    elif opt == 'only_bad':
        data_plot = data_mails.nsmallest(nb_contry, 'Sentiment')
        title = 'Hilary\'s opinion on the ' + str(nb_contry) + ' worst feelings about country'
    else:
        most_liked = data_mails.nlargest(nb_contry, 'Sentiment')
        worst_liked = data_mails.nsmallest(nb_contry, 'Sentiment')
        data_plot = pd.concat([most_liked,worst_liked])
        title = 'Hilary\'s opinion on the ' + str(nb_contry) + ' worst/best feelings about country'
        
    data_plot.sort_values(by='Sentiment', ascending=False, inplace=True)
    data_plot_copy  = data_plot.copy()
    data_plot_copy['Occurrences'] = np.log(data_plot_copy['Occurrences'])
    
    max_occurence = max(data_plot_copy.Occurrences)
    divide_max_occurence = max_occurence/4

    # Define the gradation of color
    colors = ['green' if s > max_occurence-divide_max_occurence 
         else 'palegreen' if (s < max_occurence-divide_max_occurence and s > max_occurence-2*divide_max_occurence) 
         else 'sandybrown' if (s < max_occurence-2*divide_max_occurence and s > max_occurence-3*divide_max_occurence) 
         else 'red' for s in data_plot_copy['Occurrences']]

    map_color_legend = ['Lot of occurence', 'some occurence', 'Few occurence', 'Very few occurence']
    # build the plot
    sentiment_data_plot = sns.barplot(x=data_plot.index, y='Sentiment', data=data_plot, palette=colors)
    # display a line to separate the graph.
    define_plot_legend(sentiment_data_plot,map_color_legend,title=title)
    
    if opt == None and nb_contry == 20:
        sentiment_data_plot.axvline(nb_contry - 0.5)
    sns.plt.show()
    
def define_plot_legend (plot,map_color_legend,label_y='Sentiment',title='Hilary\'s opinion on the 20 most-quoted countries'):
    '''
    Define the legend,title and label of a plot
    attribute:
        - plot : seaborn plot to modify
        - map_color_legend : label for différent color in order green,palegreen
          sandybrown and red.
        - label :ylabel
        - title : plot title
    '''
    for label in plot.get_xticklabels():
        label.set_rotation(90)
        
    plot.set(ylabel=label_y)
    plot.set_title(title)
    
    # Set color
    green_legend = mpatches.Patch(color='green', linewidth=0)
    palegreen_legend = mpatches.Patch(color='palegreen', linewidth=0)
    sandybrown_legend = mpatches.Patch(color='sandybrown', linewidth=0)
    red_legend = mpatches.Patch(color='red', linewidth=0)
    
    # Set legend
    plt.legend((green_legend, palegreen_legend, sandybrown_legend, red_legend), map_color_legend)
    
def plot_most_occurence_contry(data_mails,nb_contry = 20):
    '''
    Plot the number of country in axis, the number of occurence in ordinate
    and use 4 colors for the sentiments associate with the country. 
    attributes: 
        - data_mails : dataframe containing the data of the mails
        - nb_contry : Selection on the most representative country
    '''
    # select the data for plotting
    twenty_most_quoted_countries = data_mails.nlargest(nb_contry, 'Occurrences')

    # Define the gradation of color in order to display three variable
    colors = ['green' if s > 0.5 
         else 'palegreen' if (s < 0.5 and s > 0) 
         else 'sandybrown' if (s < 0 and s > -0.5) 
         else 'red' for s in twenty_most_quoted_countries['Sentiment']]

    countries_data_plot = sns.barplot(x=twenty_most_quoted_countries.index, y='Occurrences', data=twenty_most_quoted_countries, palette=colors)
    map_color_legend = ['Very good opinion', 'Good opinion', 'Bad opinion', 'Very bad opinion']
    define_plot_legend(countries_data_plot,map_color_legend,'Occurrences',title='Hilary\'s opinion on the ' + str(nb_contry) + ' most-quoted countries')
    sns.plt.show()
    
def plot_most_quoted_countries(data,nb_country): 
    '''
    Plot an histogram representing the nb of occurence each country appear to Hilary's mails. 
    attributes: 
        - data : dataframe sorted by the number of occurence.
        - nb_country : Selection on the most representative country
    '''
    data = data.head(nb_country)
    countries_plot = sns.barplot(x=data.index, y='Occurrences', data=data, color='lightblue')
    for label in countries_plot.get_xticklabels():
        label.set_rotation(90)
    countries_plot.set(ylabel='Occurrences')
    countries_plot.set_title('Number of occurrences of ' + str(nb_country) + ' most-quoted countries')
    sns.plt.show()