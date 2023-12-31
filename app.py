from flask import Flask, render_template, request, send_file, redirect
import json
import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import joblib
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
plt.switch_backend('Agg')
import numpy as np
from io import BytesIO
import base64
from dotenv import load_dotenv
import os
import threading

def load_classifier_and_vectorizer():
    global classifier, vectorizer
    classifier = joblib.load('./data/sentiment_classifier.pkl')
    vectorizer = joblib.load('./data/sentiment_vectorizer.pkl')

# Start loading in a separate thread
loading_thread = threading.Thread(target=load_classifier_and_vectorizer)
loading_thread.start()

app = Flask(__name__)
load_dotenv()

# Function to get Google search links
def search_links(query):
    base_url = "https://www.googleapis.com/customsearch/v1"
    api_key = os.environ['api_key']
    cx = os.environ['cx']
    params = {
        'q': query,
        'key': api_key,
        'cx': cx
    }
    Err = ""
    response = requests.get(base_url, params=params)
    if response.status_code == 429 or response.status_code == 403:
        Err = "Daily API limit reached"
        print("API limit reached.")
        return []

    results = response.json().get('items', [])
    # Extract title and link from the results and create a list of dictionaries
    links_with_titles = [[{item.get('title', ''): item.get('link', '')} for item in results], Err]

    return links_with_titles

# Function to scrape reviews using Selenium
def get_reviews(movie_url, review_type):
    page = urlopen(movie_url)
    html_bytes = page.read()
    html = html_bytes.decode("utf-8")
    emsId = re.search(r'"emsId":\s*"([^"]+)"', html).group(1)
    title = re.search(r'"titleName":\s*"([^"]+)"', html).group(1)
    title_type = re.search(r'"titleType":\s*"([^"]+)"', html).group(1)
    vanity =  re.search(r'"vanity":\s*"([^"]+)"', html).group(1)

    s_no = ""
    if title_type == "Tv":
        s_no = re.search(r'"vanity":\s*"([^"]+)"', html).group(1)

    poster_page = urlopen(f'https://www.rottentomatoes.com/{"tv" if title_type == "Tv" else "m"}/{vanity}/{f"/{s_no}/" if s_no else ""}reviews')
    soup = BeautifulSoup(poster_page, 'html.parser')
    poster_link = soup.find('img', {'data-qa': 'sidebar-poster-img'})
    if poster_link:
        poster_link = poster_link.get('src')
    else:
        poster_link = '/data/favicon.ico'
    
    hasNextPage = True
    reviews = []
    after = ""
    while hasNextPage and len(reviews) < 100:
        url = f'https://www.rottentomatoes.com/napi/{"season" if title_type == "Tv" else "movie"}/{emsId}/reviews/{review_type}?after={after}'
        response = urlopen(url).read().decode('utf-8')
        response_object = json.loads(response)
        responseArray = response_object['reviews']
        for review in responseArray:
            reviews.append(review['quote'])
        hasNextPage = response_object['pageInfo']['hasNextPage']
        if hasNextPage:
            after = response_object['pageInfo']['endCursor']
    global show_info
    show_info = [title, poster_link]
    return reviews


# Function to clean text
def clean(input_string):
    cleaned_string = re.sub(r'\s+', ' ', input_string)
    return cleaned_string.strip() 

def analyze_sentiment(reviews):
    loading_thread.join()
    
    # Clean and analyze reviews
    cleaned_reviews = [clean(phrase) for phrase in reviews]
    new_vector = vectorizer.transform(cleaned_reviews)
    predictions = classifier.predict(new_vector)

    # Return predictions
    return predictions

# Function to get polarity scores using NLTK
def get_polarity_scores(reviews):
    sia = SentimentIntensityAnalyzer()
    scores_list = []
    for entry in reviews:
        scores = sia.polarity_scores(entry)
        scores_list.append(scores)
    return scores_list

# Function to plot a bar chart
def plot_bar(data, title):
    # Extracting values for each sentiment
    neg_values = [entry['neg'] for entry in data]
    neu_values = [entry['neu'] for entry in data]
    pos_values = [entry['pos'] for entry in data]
    compound_values = [entry['compound'] for entry in data]

    # Creating positions for bars
    positions = range(len(data))

    # Plotting the stacked bars
    fig, ax = plt.subplots(figsize=(20,10))
    width = 0.7

    # Bottom bar (red)
    ax.bar(positions, neg_values, width=width, color='#961e1e', label='Negative')

    # Middle bar (gray)
    ax.bar(positions, neu_values, width=width, bottom=neg_values, color='#999', label='Neutral')

    # Top bar (green)
    ax.bar(positions, pos_values, width=width, bottom=np.array(neg_values) + np.array(neu_values), color='#015501', label='Positive')

    # Adding labels and title
    plt.xlabel('Reviews', fontdict={'fontname': 'HP Simplified', 'fontsize': 30, 'weight':'bold', 'color':'#fff'}, labelpad=20)
    plt.ylabel('Polarity Scores', fontdict={'fontname': 'HP Simplified', 'fontsize': 30, 'weight':'bold', 'color':'#fff'}, labelpad=20)
    plt.title(title, fontdict={'fontname': 'HP Simplified', 'fontsize': 40, 'weight':'bold', 'color':'#fff'}, pad=20)
    plt.yticks(fontname='HP Simplified', fontsize=24, color="#fff")

    # Remove X-axis labels
    ax.set_xticks([])
    ax.set_facecolor('#000')

    # Adding legend
    legend = plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1), prop={'family': 'HP Simplified', 'size': 32})

    average = np.mean(compound_values)
    text = f'Average Compound Score'
    avg_text = f'\n{average:.2f}'
    plt.text(0.91, 0.66, text, fontsize=20, fontname='HP Simplified', weight="bold", color="white", ha='center', va='center', transform=fig.transFigure)
    plt.text(0.91, 0.64, avg_text, fontsize=36, fontname='HP Simplified', weight="bold", color="white", ha='center', va='center', transform=fig.transFigure)

    # Set the background color
    fig.set_facecolor('#1e1e1e')
    plt.tight_layout()

    # Convert the Matplotlib figure to a Flask response
    output = BytesIO()
    FigureCanvas(fig).print_png(output)
    return base64.b64encode(output.getvalue()).decode('utf-8')

def plot_pie(data, title):
    unique, counts = np.unique(data, return_counts=True)
    print(unique, counts)

    explode = ()
    if len(unique) > 1:
        explode = (0, 0.1)
    else:
        explode = (0,)
    
    #add colors
    colors = []
    if unique[0] == "Negative":
        colors = ['#961e1e','#024d0f']
    elif unique[0] == "Positive":
        colors = ['#024d0f','#961e1e']
    
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#1e1e1e')
    ax.pie(counts, explode=explode, labels=unique, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)

    # Set custom fonts for title, labels, and autopct
    ax.set_title(title, fontdict={'family': 'HP Simplified', 'color': 'White', 'weight': 'bold', 'size': 28})
    for text in ax.texts:
        text.set_fontfamily('HP Simplified')
        text.set_fontsize('20')
        text.set_fontweight('bold')
        text.set_color('White')
        
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    plt.tight_layout()

    output = BytesIO()
    FigureCanvas(fig).print_png(output)
    return base64.b64encode(output.getvalue()).decode('utf-8')

@app.route('/data/bg.png')
def bg():
    return send_file('data/bg.png')

@app.route('/data/search-icon.svg')
def search_icon():
    return send_file('data/search-icon.svg')

@app.route('/data/favicon.ico')
def favicon():
    return send_file('data/favicon.ico')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('query')
    if query:
        try:
            search_data = search_links(query)
            if len(search_data[0]) == 0:
                search_data[1] = "No links found for your query"
            return render_template('search.html', links=search_data[0], Err=search_data[1])
        except Exception as e:
            print(e)
            return render_template('error.html', error=f"{e}")
    else:
        return redirect('/')

@app.route('/review')
def review():
    url = request.args.get('url')
    if url:
        try:       
            url = url
            user = get_reviews(url, 'user')
            print('acquired user reviews')
            critic = get_reviews(url, 'all')
            print('acquired critic reviews')
            print(show_info[0], show_info[1])
            user_pie_img, user_bar_img, critic_pie_img, critic_bar_img = '', '', '', ''

            if len(user) > 0:
                user_polarity = get_polarity_scores(user)
                print('Polarity scores calculated for user reviews')
                user_reviews = analyze_sentiment(user)
                print('Sentiment analysis completed for user reviews')
                user_bar = plot_bar(user_polarity, "User Sentiments")
                print('Bar plot generated for user reviews')
                user_pie = plot_pie(user_reviews, "User Reviews")
                print('Pie plot generated for user reviews')
                user_bar_img = f'<img src="data:image/png;base64,{user_bar}" alt="User Reviews Bar Plot">'
                user_pie_img = f'<img src="data:image/png;base64,{user_pie}" alt="User Reviews Pie Plot">'
            else:
                user_bar_img = "No User Reviews Found"
                user_pie_img = "No User Reviews Found"

            if len(critic) > 0:
                critic_polarity = get_polarity_scores(critic)
                print('Polarity scores calculated for critic reviews')
                critic_reviews = analyze_sentiment(critic)
                print('Sentiment analysis completed for critic reviews')
                critic_bar = plot_bar(critic_polarity, "Critic Sentiments")
                print('Bar plot generated for critic reviews')
                critic_pie = plot_pie(critic_reviews, "Critic Reviews")
                print('Pie plot generated for critic reviews')
                critic_bar_img = f'<img src="data:image/png;base64,{critic_bar}" alt="Critic Reviews Bar Plot">'
                critic_pie_img = f'<img src="data:image/png;base64,{critic_pie}" alt="Critic Reviews Pie Plot">'
            else:
                critic_bar_img = "No Critic Reviews Found"
                critic_pie_img = "No Critic Reviews Found"

            return render_template('review.html', user_bar=user_bar_img, user_pie=user_pie_img, critic_bar=critic_bar_img, critic_pie=critic_pie_img, title=show_info[0], url=url, poster=show_info[1])

        except Exception as  e:
            print(e)
            return render_template('error.html', error=f"{e}")
    else:
        return redirect('/')

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)