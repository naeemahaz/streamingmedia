import datetime
from flask_caching import Cache
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import Input, Output, State, dcc, html
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css']
import plotly.express as px
from dash import dcc as dcc
from datetime import datetime
from io import StringIO
import pandas as pd
import numpy as np
import tweepy as tw
import re
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
lemma = WordNetLemmatizer()
stopit = stopwords.words('english')
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
from autocorrect import Speller
from PyDictionary import PyDictionary
dictionary = PyDictionary()
nltk.download('words')
words_me = set(nltk.corpus.words.words())
from wordcloud import WordCloud
import io
import boto3
from io import StringIO
#from sagemaker import get_execution_role
import s3fs
import sys
import os
from io import StringIO
from smart_open import smart_open


import warnings
warnings.filterwarnings("ignore")


s3 = boto3.client('s3')
REGION = 'us-east-1'
ACCESS_KEY_ID = 'AKIA2K6MEWS7LIZQ2LO4'
SECRET_ACCESS_KEY = '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'
BUCKET_NAME = 'twitterdf'
cleandatame = 'twiiterdf/cleandata.csv'
filenetflix = 'twiiterdf/netflixdf.csv'
filehulu = 'twiiterdf/huludf.csv'
filetubi = 'twiiterdf/tubidf.csv'
data_key_net = 'twiiterdf/netflixdf.csv'
data_key_hul = 'twiiterdf/huludf.csv'
data_key_tub = 'twiiterdf/tubidf.csv'
data_key_clean = 'twiiterdf/cleandata.csv'
filenetflixclean = 'twiiterdf/netflixdfclean.csv'
filehuluclean = 'twiiterdf/huludfclean.csv'
filetubiclean = 'twiiterdf/tubidfclean.csv'
data_key_netclean = 'twiiterdf/netflixdfclean.csv'
data_key_hulclean = 'twiiterdf/huludfclean.csv'
data_key_tubclean = 'twiiterdf/tubidfclean.csv'

clean_pathme = 's3://{}:{}@{}/{}'.format(ACCESS_KEY_ID, SECRET_ACCESS_KEY, BUCKET_NAME, data_key_clean)
cleanall = pd.read_csv(smart_open(clean_pathme))

netclean_pathme = 's3://{}:{}@{}/{}'.format(ACCESS_KEY_ID, SECRET_ACCESS_KEY, BUCKET_NAME, data_key_netclean)
netflixexplain = pd.read_csv(smart_open(netclean_pathme))

hulclean_pathme = 's3://{}:{}@{}/{}'.format(ACCESS_KEY_ID, SECRET_ACCESS_KEY, BUCKET_NAME, data_key_hulclean)
huluexplain = pd.read_csv(smart_open(hulclean_pathme))

tubiclean_pathme = 's3://{}:{}@{}/{}'.format(ACCESS_KEY_ID, SECRET_ACCESS_KEY, BUCKET_NAME, data_key_tubclean)
tubiexplain = pd.read_csv(smart_open(tubiclean_pathme))

cleanall = pd.DataFrame({'Date': cleanall.dates, 'Network': cleanall.user_name, 'Times': cleanall.times,
                             'Results': cleanall.results})

netflixexplain = pd.DataFrame(
    {'Date': netflixexplain.date, 'Network': netflixexplain.user_name, 'Source': netflixexplain.source,
     'Text': netflixexplain.english_text, 'Words': netflixexplain.words, 'HashTags': netflixexplain.hashtagscloud,
     'Sentiment': netflixexplain.sentiment, 'Results': netflixexplain.results})


huluexplain = pd.DataFrame({'Date': huluexplain.date, 'Network': huluexplain.user_name, 'Source': huluexplain.source,
                            'Text': huluexplain.english_text, 'Words': huluexplain.words,
                            'HashTags': huluexplain.hashtagscloud, 'Sentiment': huluexplain.sentiment,
                            'Results': huluexplain.results})

tubiexplain = pd.DataFrame({'Date': tubiexplain.date, 'Network': tubiexplain.user_name, 'Source': tubiexplain.source,
                            'Text': tubiexplain.english_text, 'Words': tubiexplain.words,
                            'HashTags': tubiexplain.hashtagscloud, 'Sentiment': tubiexplain.sentiment,
                            'Results': tubiexplain.results})

negative_explain = pd.concat([netflixexplain, huluexplain, tubiexplain], ignore_index=True)

#DASHBOARD

netflixsfig = px.bar(netflixexplain, y="Source", orientation='h', title='Netflix Source Sentiments',
                     hover_data=["Words", "HashTags"], color_discrete_sequence=["#FB0D0D"])
netflixsfig.update_layout(title_font_color="#FB0D0D")
hulusfig = px.bar(huluexplain, y="Source", orientation='h', title='Hulu Source Sentiments',
                  hover_data=["Words", "HashTags"], color_discrete_sequence=["#109618"])
hulusfig.update_layout(title_font_color="#109618")
tubisfig = px.bar(tubiexplain, y="Source", orientation='h', title='Tubi Source Sentiments',
                  hover_data=["Words", "HashTags"], color_discrete_sequence=["#FE00CE"])
tubisfig.update_layout(title_font_color="#FE00CE")


netflixfig = px.bar(netflixexplain, y="Results", orientation='h', title='Netflix Sentiments',
                    hover_data=["Source", "Words", "HashTags"], color_discrete_sequence=["#FB0D0D"])
netflixfig.update_layout(title_font_color="#FB0D0D")
hulufig = px.bar(huluexplain, y="Results", orientation='h', title='Hulu Sentiments',
                 hover_data=["Source", "Words", "HashTags"], color_discrete_sequence=["#109618"])
hulufig.update_layout(title_font_color="#109618")
tubifig = px.bar(tubiexplain, y="Results", orientation='h', title='Tubi Sentiments',
                 hover_data=["Source", "Words", "HashTags"], color_discrete_sequence=["#FE00CE"])
tubifig.update_layout(title_font_color="#FE00CE")

username_fig = px.histogram(cleanall, x="Results", color="Network", title='Streaming Media Twitter Sentiments',
                            color_discrete_sequence=["#FB0D0D", "#109618", "#FE00CE"])
username_fig.update_layout(barmode='group')

usercount_fig = px.histogram(cleanall, x="Results", color="Network", marginal="rug",
                             title='Streaming Media Twitter', color_discrete_sequence=["#FB0D0D", "#109618", "#FE00CE"])

cleanall.Date = pd.to_datetime(cleanall.Date)
cleanall.Date.sort_values().index
df_by_date = cleanall.iloc[cleanall.Date.sort_values().index]
cleanall_by_date = df_by_date.groupby(['Date', 'Network']).Date.count()

cleanallme = cleanall[['Date', 'Times', 'Network', 'Results']]

netflix_me = cleanallme.query('Network == ["Netflix"]')
hulu_me = cleanallme.query('Network == ["Hulu"]')
tubi_me = cleanallme.query('Network == ["Tubi"]')

cleanallme.Date = pd.to_datetime(cleanallme.Date)
cleanallme.Date.sort_values().index
cleanallme_df_by_date = cleanallme.iloc[cleanallme.Date.sort_values().index]
cleanallme_line = cleanallme_df_by_date.groupby('Date').Date.count()
cleanallme_fig = px.line(cleanallme_line, y='Date', title='Twitter Usage for All Network')

netflixhashwords = netflixexplain['HashTags'].astype(str)
netflix_wordcloud = WordCloud(background_color='white', height=275, width=400).generate(' '.join(netflixhashwords))
fig_netflix = px.imshow(netflix_wordcloud, title="Netflix Hashtags")
fig_netflix.update_xaxes(visible=False)
fig_netflix.update_yaxes(visible=False)
fig_netflix.update_layout(title_font_color="#FB0D0D")

huluhashwords = huluexplain['HashTags'].astype(str)
hulu_wordcloud = WordCloud(background_color='white', height=275, width=400).generate(' '.join(huluhashwords))
fig_hulu = px.imshow(hulu_wordcloud, title="Hulu Hashtags")
fig_hulu.update_xaxes(visible=False)
fig_hulu.update_yaxes(visible=False)
fig_hulu.update_layout(title_font_color="#109618")

tubihashwords = tubiexplain['HashTags'].astype(str)
tubi_wordcloud = WordCloud(background_color='white', height=275, width=400).generate(' '.join(tubihashwords))
fig_tubi = px.imshow(tubi_wordcloud, title="Tubi Hashtags")
fig_tubi.update_xaxes(visible=False)
fig_tubi.update_yaxes(visible=False)
fig_tubi.update_layout(title_font_color="#FE00CE")

netflixwords = netflixexplain['Words'].astype(str)
netflixwordscloud = WordCloud(background_color='white', height=275, width=400).generate(' '.join(netflixwords))
fig_netflixwords = px.imshow(netflixwordscloud, title="Netflix Twitter Words")
fig_netflixwords.update_xaxes(visible=False)
fig_netflixwords.update_yaxes(visible=False)
fig_netflixwords.update_layout(title_font_color="#FB0D0D")

huluwords = huluexplain['Words'].astype(str)
huluwordscloud = WordCloud(background_color='white', height=275, width=400).generate(' '.join(huluwords))
fig_huluwords = px.imshow(huluwordscloud, title="Hulu Twitter Words")
fig_huluwords.update_xaxes(visible=False)
fig_huluwords.update_yaxes(visible=False)
fig_huluwords.update_layout(title_font_color="#109618")

tubiwords = tubiexplain['Words'].astype(str)
tubiwordscloud = WordCloud(background_color='white', height=275, width=400).generate(' '.join(tubiwords))
fig_tubiwords = px.imshow(tubiwordscloud, title="Tubi Twitter Words")
fig_tubiwords.update_xaxes(visible=False)
fig_tubiwords.update_yaxes(visible=False)
fig_tubiwords.update_layout(title_font_color="#FE00CE")





negative_explain[['Date', 'Network', 'Text',  'Results']]
neg_twitter = negative_explain[negative_explain['Results'] == 'Negative']

table_fig = go.Figure(data=[go.Table(
    header=dict(values=list(neg_twitter.columns),
                line_color='black',
                fill_color='darkslategray',
                font_color='white',
                align='left'),
    cells=dict(values=[neg_twitter.Network, neg_twitter.Date, neg_twitter.Text, neg_twitter.Results],
               line_color='black',
               fill_color='white',
               font_color='black',
               align='left'))
])

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})
app.config.suppress_callback_exceptions = True
timeout = 10

server = app.server

app.layout = html.Div([
    html.Div(id='flask-cache-memoized-children'),
    dcc.Tabs([
        dcc.Tab(label='Social Media Live Twitter Results', children=[
            dbc.Row(dbc.Col(html.H4('Streaming TV Media Overall - (200 tweets per Network)'))),
            dbc.Row([dbc.Col([dcc.Graph(figure=cleanallme_fig, style={'height': 600, 'width': 600}), ],
                             width={'size': 4, 'offset': 0, 'order': 1}),
                     dbc.Col([dcc.Graph(figure=usercount_fig, style={'height': 600, 'width': 600}), ],
                             width={'size': 4, 'offset': 0, 'order': 2}),
                     dbc.Col([dcc.Graph(figure=username_fig, style={'height': 600, 'width': 600}), ],
                             width={'size': 4, 'offset': 0, 'order': 3}), ]),

            dbc.Row(dbc.Col(html.H4('Streaming TV Media Twitter Source'))),
            dbc.Row([dbc.Col([dcc.Graph(figure=netflixsfig, style={'height': 600, 'width': 600}), ],
                             width={'size': 4, 'offset': 0, 'order': 1}),
                     dbc.Col([dcc.Graph(figure=hulusfig, style={'height': 600, 'width': 600}), ],
                             width={'size': 4, 'offset': 0, 'order': 2}),
                     dbc.Col([dcc.Graph(figure=tubisfig, style={'height': 600, 'width': 600}), ],
                             width={'size': 4, 'offset': 0, 'order': 3}), ]),

            dbc.Row(dbc.Col(html.H4('Streaming TV Media Sentiments'))),
            dbc.Row([dbc.Col([dcc.Graph(figure=netflixfig, style={'height': 600, 'width': 600}), ],
                             width={'size': 4, 'offset': 0, 'order': 1}),
                     dbc.Col([dcc.Graph(figure=hulufig, style={'height': 600, 'width': 600}), ],
                             width={'size': 4, 'offset': 0, 'order': 2}),
                     dbc.Col([dcc.Graph(figure=tubifig, style={'height': 600, 'width': 600}), ],
                             width={'size': 4, 'offset': 0, 'order': 3}), ]),

            dbc.Row(dbc.Col(html.H4('Streaming TV Media Hashtags'))),
            dbc.Row([dbc.Col([dcc.Graph(figure=fig_netflix, style={'height': 600, 'width': 600}), ],
                             width={'size': 4, 'offset': 0, 'order': 1}),
                     dbc.Col([dcc.Graph(figure=fig_hulu, style={'height': 600, 'width': 600}), ],
                             width={'size': 4, 'offset': 0, 'order': 2}),
                     dbc.Col([dcc.Graph(figure=fig_tubi, style={'height': 600, 'width': 600}), ],
                             width={'size': 4, 'offset': 0, 'order': 3}), ]),

            dbc.Row(dbc.Col(html.H4('Streaming TV Media Tweets Text'))),
            dbc.Row([dbc.Col([dcc.Graph(figure=fig_netflixwords, style={'height': 600, 'width': 600}), ],
                             width={'size': 4, 'offset': 0, 'order': 1}),
                     dbc.Col([dcc.Graph(figure=fig_huluwords, style={'height': 600, 'width': 600}), ],
                             width={'size': 4, 'offset': 0, 'order': 2}),
                     dbc.Col([dcc.Graph(figure=fig_tubiwords, style={'height': 600, 'width': 600}), ],
                             width={'size': 4, 'offset': 0, 'order': 3}), ]),

            dbc.Row(dbc.Col(html.H4('Streaming TV Media Negative Tweets'))),
            # dbc.Row([dbc.Col([dcc.Graph(figure=netflixsfig, style={'height':300, 'width':400}),],width={'size': 4, 'offset': 0, 'order': 1}),
            # dbc.Col([dcc.Graph(figure=netflixsfig, style={'height':300, 'width':400}), ],width={'size': 4, 'offset': 0, 'order': 2}),
            dbc.Col([dcc.Graph(figure=table_fig, style={'height': 800, 'width': 1600}), ],
                    width={'size': 4, 'offset': 0, 'order': 3})
        ]),
        dcc.Tab(label='Resume', children=[
            dbc.Row(dbc.Col(html.H4('Naeemah Aliya Small'))),
            dbc.Row(dbc.Col(dcc.Link('LinkedIn: http://www.linkedin.com/in/naeemah-small',
                                     href='http://www.linkedin.com/in/naeemah-small'))),
            dbc.Row(dbc.Col(dcc.Link('Github: https://github.com/naeemahaz', href='https://github.com/naeemahaz'))),
            dbc.Row(dbc.Col(html.H6('Email: naeemahaz@gmail.com'))),
            dbc.Row(dbc.Col(html.H6('Phone: 520-405-2724'))),
            dbc.Row(dbc.Col(html.H1())),
            dbc.Row(dbc.Col(html.P(
                'I believe that my greatest strength is the ability to solve problems quickly and efficiently. I can see any given situation from multiple perspectives, making me uniquely qualified to execute my work even under challenging conditions. That problem-solving skill allows me to be a better communicator. I am just as comfortable speaking to senior executives and junior team members. I think my objectivity will make me a great asset to the company.'))),
            dbc.Row(dbc.Col(html.P(
                'An Analyst with a track record of success in collecting, analyzing, and manipulating large datasets. Expert ability to analyze and interpret complex data models to identify critical operational impacts to drive positive results. My advanced leadership skills allow me to establish strong partnerships with cross-functional teams to support continuous process improvement. Technically proficient in multiple systems, tools, and applications for data management and analysis.'))),
            dbc.Row(dbc.Col(html.H5('Areas of Excellence'))),
            dbc.Row(dbc.Col(html.P(
                'Qualitative and Quantitative Research • Data Planning & Analysis • Business Analysis • Strategic & Tactical Planning • Business Process Optimization • Deep Learning • Recommender • Products, Services, Price, and Sales Analytics • Growth Analytics • Cross-Functional Collaboration • Process Improvement •Database Management • Predictive Modeling/Forecasting • Executive Presentations • Program Analysis • Process Analysis & Restructuring • Technically Proficient'))),
            dbc.Row(dbc.Col(html.H1())),
            dbc.Row(dbc.Col(html.H5('Achievements'))),
            dbc.Row(dbc.Col(html.P(
                '* Perform data collection, cleaning, validation, and reporting tasks for digital commerce systems and applications. Design and develop advanced analytical data mining/machine learning algorithms and predictive models. Research, design, develop, implement, and support decision science models.  Provide analytical consultation, predictive modeling, and solutions to achieve business goals and objectives. Perform exploratory data analysis to find patterns and trends in business applications. Develop metrics to measure the incremental benefit/costs of competing analytic models. Validate rational solutions to determine overall effectiveness—conduct segmentation analysis for risk management and targeted marketing purposes. I performed data mining using advanced data analysis techniques to identify performance patterns and trends.  Using Seaborn, Pandas, Numpy, Spacy, Recordlinkage, Genism, Nltk, Textblob, Pickle, Scipy, and Sklearn in Python. '))),
            dbc.Row(dbc.Col(html.P(
                '* I assisted in new product development to ensure proper coverage and profitability, performing pro forma analysis on pricing strategies and evaluating product performance by leveraging insights to optimize pricing strategies and business development strategies using Python at JP Morgan and Chase. In addition, by documenting program bugs and creating improvements to the new product, the bank continued to generate more priority programs. '))),
            dbc.Row(dbc.Col(html.P(
                '* Worked closely with the vice president to develop a strategy for promoting new credit card products and services to targeted customers and other programs to increase company revenue using Machine Learning using Python at JP Morgan and Chase. '))),
            dbc.Row(dbc.Col(html.P(
                '* Selected to participate in a unique project to promote the utilization of a new authentication product/service; recommended introducing the new service to loyal customers to enhance enrollment using Qualitative and Quantitative Research. Created a training video on promoting authentication service for all team members at JP Morgan and Chase. '))),
            dbc.Row(dbc.Col(html.H1())),
            dbc.Row(dbc.Col(html.H5('Published:'))),
            dbc.Row(dbc.Col(dcc.Link('Clean Your Data in Seconds with This R Function',
                                     href='https://datascienceplus.com/clean-your-data-in-seconds-with-this-r-function'))),
            dbc.Row(dbc.Col(dcc.Link('Cleaning & Modifying A Dataframe – Python',
                                     href='https://datascienceplus.com/cleaning-modifying-a-dataframe-python/'))),
            dbc.Row(dbc.Col(
                dcc.Link('Who Will Get Promoted?', href='https://www.kaggle.com/subzeroheart/who-will-get-promoted'))),
            dbc.Row(dbc.Col(dcc.Link('The Non-Bias Hiring Model using the Jaro-Winkler algorithm in Python',
                                     href='https://medium.com/@naeemahaz/the-non-bias-hiring-model-423199b7ca2c'))),
            dbc.Row(dbc.Col(dcc.Link('How To Create A Multiple Language Dictionary Using A Pipeline',
                                     href='https://datascienceplus.com/how-to-create-a-multiple-language-dictionary-using-a-pipeline'))),
            dbc.Row(dbc.Col(dcc.Link('How to View Live Tweets in an Excel Dashboard using Python & VBA',
                                     href='https://medium.com/@naeemahaz/how-to-view-live-tweets-in-an-excel-dashboard-using-python-vba-6fb537361565'))),
            dbc.Row(dbc.Col(html.H1())),
            dbc.Row(dbc.Col(html.H5('Technical Skills'))),
            dbc.Row(dbc.Col(html.P(
                'C, Python 3.0, R, SQL, Spark, Microsoft Office, Tableau, Power BI, Azure, SQL, VBA, Visio, Xcelsius, Cognos, Toad, Oracle, Jira, Content Management, AS400, Azure Machine Learning, Jupyter Notebook, Agile Methodologies '))),

        ]

                )
    ])
])

@app.callback(
    Output('flask-cache-memoized-children', 'children'),
    Input('flask-cache-memoized-dropdown', 'value'))
@cache.memoize(timeout=timeout)  # in seconds
def render(value):
    current_time = datetime.datetime.now().strftime('%H:%M:%S')
    return f'Selected "{value}" at "{current_time}"'


if __name__ == '__main__':
    app.run_server(debug=False)  # deployment
    #app.run_server(debug=False, host="0.0.0.0", port=8080)  # deployment