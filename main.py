from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
#import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

# general syntax for url on finviz.com
finviz_url = 'https://finviz.com/quote.ashx?t='
# FLAG
#tickers = ['FB','AMZN','AAPL','GOOG','MSFT']
# Auto GeneralMotor, Fiat, Toyota, Tesla
tickers = ['GM','FCAU','TM','TSLA']
# AT&T, Verizon, T-mobile
#tickers = ['T','VZ','TMUS']

news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)

    html = BeautifulSoup(response,'html')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

'''
data_fb = news_tables['FB']
# find all the rows with element tr
rows_fb = data_fb.findAll('tr')

for index, row in enumerate(rows_fb):
    # the title is in the tag <a>...</a>
    # <a class="tab-link-news" href="https://finance.yahoo.com/video/expect-blockbuster-earnings-week-ahead-191216677.html" target="_blank">What to expect from pivotal earnings week ahead</a>
    title = row.a.text
    # the timestamp is in the tag <td>...</td>
    # <tr><td align="right" width="130">03:47PM  </td>
    timestamp = row.td.text
    print(timestamp + ' ' + title)
'''

parsed_data = []
for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        title = row.a.text # or row.a.get_text()
        # the timestamp may look like
        # 09:06AM or Jul-23-20 08:21PM -> split with ' '
        timestamp = row.td.text.split(' ')
        if len(timestamp) == 1:
            time = timestamp[0][0:7]
        else:
            date = timestamp[0]
            time = timestamp[1][0:7]

        parsed_data.append([ticker, date, time, title])

"""
for data in parsed_data:
    print(data)
"""

df = pd.DataFrame(parsed_data, columns=['ticker','date','time','title'])
vader = SentimentIntensityAnalyzer()
# vader gives out 4 scores: compound, negative, neutral, positive
# the overall attitude and component in 3 different attitude
# this lambda function returns the compound score for any title
f = lambda title: vader.polarity_scores(title)['compound']
# apply the lambda function to the title and save the result in a new column
df['compound'] = df['title'].apply(f)
# convert the date(str) to a recognizable date
df['date'] = pd.to_datetime(df.date).dt.date


mean_df = df.groupby(['ticker','date']).mean()
# 3 columns(ticker, date, compound) x 6 rows(date)
mean_df = mean_df.unstack()
# 1 row (score) x 6 columns(date)
mean_df = mean_df.xs('compound', axis='columns').transpose()
#mean_df.plot(kind='bar')

# mean_df.plot(figsize=(10, 8))
fig1 = plt.figure(figsize=(10, 8))
fig1.plot(mean_df)
plt.title("Sentiment Analysis", fontsize=24)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Compound", fontsize=14)
plt.legend(title='Brand', labels=['General Motors','Fiat Chrysler','Toyota','Tesla'],
           fancybox=True, framealpha=1, shadow=True, borderpad=1,
           loc='best',)
plt.tick_params(axis="both", labelsize=14)
plt.tick_params(axis="x", rotation=30)
plt.show()