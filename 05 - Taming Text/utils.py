# IMPORT SECTION #
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nltk.data
import numpy as np
import pandas as pd
import pycountry
import random
import re
import seaborn as sns

from collections import Counter
from gensim import models, corpora
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from os import path
from os.path import exists
from PIL import Image
# END: IMPORT SECTION # 



# GLOBAL VARIABLES AND CONSTANTS #
PORTER_STEMMER = PorterStemmer()
REGEX_TOKENIZER = RegexpTokenizer(r'\w+')
SENTENCES_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')
SID = SentimentIntensityAnalyzer()
SNOWBALL_STEMMER = SnowballStemmer("english")
SPECIFIC_STOP_WORDS = ['re', 'pm', 'will', 'said', 'say', 'Mr', 'also']
WORDNET_LEMMATIZER = WordNetLemmatizer()
# END: GLOBAL VARIABLES AND CONSTANTS #


# FUNCTIONS #
def do_stemming_words(stemmer, words):
    '''
    This function generates a string while applying a stemmer on each word of an array.
    
    Parameters
        - stemmer: stemmer to apply on each word
        - words  : array of strings on which stemmer will be applied
    
    Return
        - text: string with stemmed words
    '''

    text = ''
    for w in words:
        text += stemmer.stem(w)
    return text


def process_email_content(email):
    '''
    This function cleans an email content.

    Parameters
        - email: email to be cleaned

    Return
        - final_text: cleaned text
    '''

    email_content = str(email['ExtractedBodyText'])

    tokens = REGEX_TOKENIZER.tokenize(email_content)
    
    filtered_words = [word for word in tokens if word not in stopwords.words('english')]
    filtered_text = generate_raw_text(data=filtered_words)

    for word in SPECIFIC_STOP_WORDS: 
        filtered_text = re.sub(r'\b%s\b' % word, '', filtered_text, flags=re.IGNORECASE)
    
    wl_text = WORDNET_LEMMATIZER.lemmatize(filtered_text)
    
    final_text = do_stemming_words(stemmer=SNOWBALL_STEMMER, words=wl_text)
    
    return final_text


def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    '''
    This function defines a color function.

    See: https://amueller.github.io/word_cloud/auto_examples/a_new_hope.html
    '''

    return "hsl(0, 0%%, %d%%)" % random.randint(40, 60)


def generate_word_cloud(text, img_name='envelope.png', max_words=1000, width=900, height=900, dpi=400, file_name=None):
    '''
    This function generates and saves (if asked) a word cloud given a set of words.

    Parameters
        - text     : text to study
        - img_name : name of the picture used for the word cloud
        - max_words: limit of displayed words
        - width    : width of the picture
        - height   : height of the picture
        - dpi      : quality of the picture
        - file_name: name of the file if picture must be saved (picture is saved in ./images/) 
    
    Return
        - plt: Image of word cloud
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


def check_if_country_in_text(country, text):
    '''
    This function checks if a given country is mentionned in a given text (search according name and official name only).

    Parameters
        - country: country to find in text
        - text   : text on which search is performed

    Return
        - boolean value (True if country was found, False otherwise)
    '''

    return ((str.lower(country.name) in text) or (hasattr(country, 'official_name') and (str.lower(country.official_name) in text)))


def count_country_occurrences(country, text):
    '''
    This function counts the number of occurrences of a specific country in a given text (count according name and official name only).

    Parameters
        - country: name of the country
        - text   : text on which count is performed
        
    Return
        - nb_occurrences: Number of occurrences of the country in the text
    '''

    nb_occurrences = 0
    nb_occurrences += text.count(str.lower(country.name))
    if hasattr(country, 'official_name'):
        nb_occurrences += text.count(str.lower(country.official_name))
    return nb_occurrences


def count_countries_occurrences(text):
    '''
    This function counts the number of occurrences of all countries in a given text.

    Parameters
        - text: text to study
        
    Return
        - countries: Counter containing, for each country, the number of apparition in the text
    '''

    lower_text = str.lower(text)
    countries = Counter()
    for country in pycountry.countries:
        nb_occurrences = count_country_occurrences(country, lower_text)
        countries[country.name] = nb_occurrences
    return countries

    
def get_wordnet_tag_type(tag):
    '''
    This function returns the type associated to a given tag.

    Parameters
        - tag: tag to be analyzed

    Return
        - tag type (None by default)
    '''

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
    This function associates a POSITIVE / NEGATIVE feeling given a set of words, according of scores returned by SentiWordNet.
    
    Parameters
        - tokens: text to study
        
    Return
        - types: Counter containing the number of global positive/negative scores of a given text
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
    This function associates a POSITIVE / NEGATIVE feeling given a set of words, according of scores returned by Vader.

    Parameters
        - email_content: text to study
        
    Return
        - types: Counter containing the number of global positive/negative scores of a given text
    '''

    types = Counter({'Positive': 0, 'Negative': 0})
    tokens = SENTENCES_DETECTOR.tokenize(email_content.strip())
    for sentence in tokens:
        scores = SID.polarity_scores(sentence)
        types['Positive'] += scores['pos']
        types['Negative'] += scores['neg']
    return types


def retrieve_email_sentiment(email, analyzer='sentiwordnet'):
    '''
    This function retrieves the sentiment associated to an email.

    Parameters
        - email   : email to analyze (row of DataFrame)
        - analyzer: type of analyzer (sentiwordnet or Vader; by default, sentiwordnet is used)

    Return
        - email: modified email containing sentiment
    '''

    email_content = str(email['MetadataSubject']) + ' ' + str(email['ExtractedBodyText'])
    
    if(analyzer == 'Vader'):
        types = get_vader_scores(email_content)
    else:
        tokens = REGEX_TOKENIZER.tokenize(email_content)
        types = get_sentiwordnet_scores(tokens)
    
    # Manual verification (max is instable in case of equal values)
    if types['Positive'] > abs(types['Negative']):
        email['Type'] = 'Positive'
    elif abs(types['Negative']) > types['Positive']:
        email['Type'] = 'Negative'
    else:
        email['Type'] = 'Neutral'
        
    return email


def get_countries_sentiment(emails):
    '''
    This function retrieves the sentiment associated to all countries, and according of sentiment of emails in which it is quoted.

    Parameters
        - emails: set of emails to analyze

    Return
        - countries_sentiment: Counter containing sentiment score for each countey (positive score <=> positive feeling / negative score <=> bad feelind / 0 <=> neutral)
        '''

    countries_sentiment = Counter()

    # Loop on emails
    for index, email in emails.iterrows():
        email_content = str(email['MetadataSubject']) + ' ' + str(email['ExtractedBodyText'])
        lower_email_content = str.lower(email_content)

        # Loop on countries
        for country in pycountry.countries:
            is_in_text = check_if_country_in_text(country, lower_email_content)
            
            # Update counter if country is in current email
            if(is_in_text == True):
                if email['Type'] == 'Positive':
                    countries_sentiment[country.name] += 1
                elif email['Type'] == 'Negative':
                    countries_sentiment[country.name] -= 1
                else:
                    countries_sentiment[country.name] += 0

    return countries_sentiment


def plot_sentiment_by_country(data_mails, opt, nb_country = 20):
    '''
    This function plots sentiment of countries (countries are filtered by sentiment).
    This function uses 4 colors to evaluate the occurrences of given countries.
    Abscissa: countries / Ordinate: sentiment / Color: occurrences

    Parameters
        - opt        : allow to select the option of the graph
            - 'only_good' = keep only good feeling
            - 'only_bad'  = keep only bad feeling
            - <empty>     = concatenate good and bad feeling in same plot
        - data_mails : DataFrame containing the data (emails)
        - nb_country : selection of the most representative countries (by default, 20)
    '''

    # Selection of data to be used for the plot
    if opt == 'only_good':
        data_plot = data_mails.nlargest(nb_country, 'Sentiment')
        title = 'Hilary\'s opinion for the ' + str(nb_country) + ' most-preferred countries'
    elif opt == 'only_bad':
        data_plot = data_mails.nsmallest(nb_country, 'Sentiment')
        title = 'Hilary\'s opinion for the ' + str(nb_country) + ' less-preferred countries'
    else:
        most_liked = data_mails.nlargest(nb_country, 'Sentiment')
        worst_liked = data_mails.nsmallest(nb_country, 'Sentiment')
        data_plot = pd.concat([most_liked,worst_liked])
        title = 'Hilary\'s opinion for the ' + str(nb_country) + ' most- and less-preferred countries'
    
    data_plot.sort_values(by='Sentiment', ascending=False, inplace=True)
    data_plot_copy  = data_plot.copy()
    # We use log because of the distribution of data
    data_plot_copy['Occurrences'] = np.log(data_plot_copy['Occurrences'])
    
    max_occurence = max(data_plot_copy.Occurrences)
    divided_max_occurence = max_occurence/4

    # Definition of the gradation of color
    colors = ['green' if s > max_occurence-divided_max_occurence 
         else 'palegreen' if (s < max_occurence-divided_max_occurence and s > max_occurence-2*divided_max_occurence) 
         else 'sandybrown' if (s < max_occurence-2*divided_max_occurence and s > max_occurence-3*divided_max_occurence) 
         else 'red' for s in data_plot_copy['Occurrences']]
    map_color_legend = ['Lot of occurence', 'some occurence', 'Few occurence', 'Very few occurence']
    
    # Creation of the plot
    sentiment_data_plot = sns.barplot(x=data_plot.index, y='Sentiment', data=data_plot, palette=colors)
    
    # Display of a line to separate the graph (static line, for delimitation of 20 most- and less-preferred countries)
    define_plot_legend(sentiment_data_plot,map_color_legend,title=title)
    
    if opt == None and nb_country == 20:
        sentiment_data_plot.axvline(nb_country - 0.5)

    sns.plt.show()


def define_plot_legend (plot, map_color_legend, label_y = 'Sentiment', title = 'Hilary\'s opinion for the 20 most-quoted countries'):
    '''
    This function defines the legend, title and label of a plot.
    
    Parameters
        - plot            : seaborn plot to modify
        - map_color_legend: label for different colors in order green,p alegreen, sandybrown and red.
        - label_y         : Y label (by default, Sentiment)
        - title           : title (by default, Hilary's opinion on the 20 most-quoted countries)
    '''

    for label in plot.get_xticklabels():
        label.set_rotation(90)
        
    plot.set(ylabel=label_y)
    plot.set_title(title)
    
    # Set of color
    green_legend = mpatches.Patch(color='green', linewidth=0)
    palegreen_legend = mpatches.Patch(color='palegreen', linewidth=0)
    sandybrown_legend = mpatches.Patch(color='sandybrown', linewidth=0)
    red_legend = mpatches.Patch(color='red', linewidth=0)
    
    # Set of legend
    plt.legend((green_legend, palegreen_legend, sandybrown_legend, red_legend), map_color_legend)
    

def plot_countries_by_occurrences_and_sentiment(data_mails, nb_country = 20):
    '''
    This function plots the occurrences of countries.
    This function uses 4 colors to evaluate the possible sentiment of given countries.
    Abscissa: countries / Ordinate: occurrences / Color: sentiment

    Parameters 
        - data_mails: DataFrame containing the data (emails)
        - nb_country: selection on the most representative countries (20 by default)
    '''

    # We select data for plotting
    twenty_most_quoted_countries = data_mails.nlargest(nb_country, 'Occurrences')

    # We define the gradation of colors in order to display three variables
    colors = ['green' if s > 0.5 
         else 'palegreen' if (s < 0.5 and s > 0) 
         else 'sandybrown' if (s < 0 and s > -0.5) 
         else 'red' for s in twenty_most_quoted_countries['Sentiment']]

    # We create and display graph
    countries_data_plot = sns.barplot(x=twenty_most_quoted_countries.index, y='Occurrences', data=twenty_most_quoted_countries, palette=colors)
    map_color_legend = ['Very good opinion', 'Good opinion', 'Bad opinion', 'Very bad opinion']
    define_plot_legend(countries_data_plot,map_color_legend,'Occurrences',title='Hilary\'s opinion for the ' + str(nb_country) + ' most-quoted countries')
    sns.plt.show()
    

def plot_most_quoted_countries(data, nb_country): 
    '''
    This function plots an histogram representing the number of occurrences of most-quoted countries.

    Parameters
        - data       : DataFrame sorted by the number of occurrences
        - nb_country : selection on the most representative countries
    '''

    data = data.head(nb_country)
    countries_plot = sns.barplot(x=data.index, y='Occurrences', data=data, color='hotpink')
    for label in countries_plot.get_xticklabels():
        label.set_rotation(90)
    countries_plot.set(ylabel='Occurrences')
    countries_plot.set_title('Number of occurrences of ' + str(nb_country) + ' most-quoted countries')
    sns.plt.show()


# Following code was taken from StackOverflow (credits to alvas) and modified accordingly
def create_corpus(content, processed=False):
    '''
    This function create a corpus given a content.

    Parameters
        - content  : content to be used to create the corpus (string)
        - processed: boolean which indicates if content was already processed or not (False by default)

    Return
        - corpus : created corpus for given content
        - id2word: dictionary containing ids and associated words
    '''

    # Creation of the dictionary (if content was not processed) using all the retrieved sentences
    if not processed:
        all_text_array = [[word for word in sentence.lower().split() if word not in stopwords.words('english')] for sentence in content]
    else:
        all_text_array = content

    dictionary = corpora.Dictionary(all_text_array)
    
    # Creation of links between ids and words for better readability
    id2word = {}
    for word in dictionary.token2id:    
        id2word[dictionary.token2id[word]] = word

    # Creation of the corpus
    corpus = [dictionary.doc2bow(text) for text in all_text_array]

    return corpus, id2word


def create_lda_model(corpus, id2word, nb_topics=5):
    '''
    This function creates a LDA model given a corpus.

    Parameters
        - corpus : corpus to be used
        - id2word: dictionary containing ids and associated words

    Return
        - lda: LDA model
    '''

    lda = models.LdaModel(corpus, id2word=id2word, num_topics=nb_topics)
    return lda


def get_text_without_stopwords(data): 
    '''
    This function creates an array containing arrays of text (one array per discussion); this array can be used when creating a corpus.
    
    Parameters
        - data: data containing contents to be processed
    
    Return
        - all_text_array: array containing the set of words for each discussion
    '''

    all_text_array = []
    for content in data:
        all_text_array += [[word for word in sentence.lower().split() if word not in stopwords.words('english')] 
                           for sentence in SENTENCES_DETECTOR.tokenize(content.strip())]
    
    return all_text_array
# END: FUNCTIONS #
  