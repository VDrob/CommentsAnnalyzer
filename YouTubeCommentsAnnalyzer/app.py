from flask import Flask, render_template, request
from googleapiclient.discovery import build
from transformers import pipeline
import re  # Імпортуємо модуль для регулярних виразів
import matplotlib.pyplot as plt
import io
import base64
import matplotlib

matplotlib.use('Agg')
app = Flask(__name__)

# Конфігурація API ключа для Google API
api_key = "AIzaSyC1NkUA6QGOT1et5dA_INirKB5IClYJS1c"  # Замініть на дійсний API ключ

# Створення клієнта для Google API
youtube = build('youtube', 'v3', developerKey=api_key)

# Створення моделі для аналізу коментарів
model = pipeline('sentiment-analysis')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Отримання посилання на відео з форми
        video_url = request.form['video_id']

        # Витягування video_id з URL
        video_id = extract_video_id(video_url)
        print(f"Video ID to analyze: {video_id}")
        if video_id:
            # Виклик функції для аналізу коментарів
            _, stats = analyze_comments(video_id)

            # Генерація графіка
            graph_url = create_graph(stats)
            print(f"Graph URL: {graph_url}")
            # Відображення результатів аналізу
            return render_template('results.html', graph_url=graph_url, stats=stats)
        else:
            return render_template('results.html', graph_url=None, stats={"total": 0, "positive": 0, "negative": 0})
    
    return render_template("index.html")

def extract_video_id(url):
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', url)
    if match:
        return match.group(1)
    return None

def analyze_comments(video_id):
    comments = []
    next_page_token = None
    stats = {'total': 0, 'positive': 0, 'negative': 0}
    
    while True:
        try:
            # Виклик API для отримання коментарів до відео
            comments_response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=100,  # Максимальна кількість коментарів на сторінці
                pageToken=next_page_token
            ).execute()

            # Додаємо нові коментарі до загального списку
            comments += comments_response.get('items', [])
            
            # Оновлюємо токен для наступної сторінки
            next_page_token = comments_response.get('nextPageToken')

            # Якщо немає більше сторінок, виходимо з циклу
            if not next_page_token:
                break
        except Exception as e:
            print(f"Error fetching comments: {e}")
            return [{"text": "Error fetching comments.", "sentiment": None}], stats
    
    if not comments:
        return [{"text": "No comments found.", "sentiment": None}], stats

    # Аналіз коментарів за допомогою моделі
    sentiments = []
    for comment in comments:
        text = comment['snippet']['topLevelComment']['snippet']['textDisplay']
        
                # Пропускаємо коментарі довші за 512 символів
        if len(text) > 512:
            print("Skipping comment due to length.")
            continue
        if re.search(r'http[s]?://', text):
            print("Skipping comment due to link.")
            continue


        try:
            sentiment = model(text)
            sentiments.append({'text': text, 'sentiment': sentiment[0]['label']})
            stats['total'] += 1
            if sentiment[0]['label'] == 'POSITIVE':
                stats['positive'] += 1
            elif sentiment[0]['label'] == 'NEGATIVE':
                stats['negative'] += 1
        except Exception as e:
            print(f"Error analyzing comment: {e}")
            sentiments.append({'text': text, 'sentiment': "Error analyzing comment."})
    print(f"Fetched {len(comments)} comments.")
    return sentiments, stats

def create_graph(stats):
    print(f"Stats: {stats}")
    labels = ['Positive', 'Negative']
    sizes = [stats['positive'], stats['negative']]
    
    # Перевірка на нульові значення
    if all(size == 0 for size in sizes):
        sizes = [1, 1]  # Уникнути NaN, якщо немає коментарів
        labels = ['No Positive', 'No Negative']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('Sentiment Analysis')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

if __name__ == '__main__':
    app.run(debug=True)