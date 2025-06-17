import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from transformers import pipeline
import matplotlib.pyplot as plt
import torch  # Required for checking GPU availability

# Load API key
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# Set device: 0 for GPU if available, -1 for CPU
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to {'GPU' if device == 0 else 'CPU'}")

# ========== Helper Functions ==========

def extract_video_id(url):
    query = urlparse(url)
    return parse_qs(query.query).get("v", [None])[0]

def get_video_metadata(video_id):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    request = youtube.videos().list(part="snippet,statistics", id=video_id)
    response = request.execute()
    item = response["items"][0]
    snippet = item["snippet"]
    stats = item["statistics"]
    return {
        "title": snippet["title"],
        "channel": snippet["channelTitle"],
        "upload_date": snippet["publishedAt"],
        "likes": int(stats.get("likeCount", 0)),
        "views": int(stats.get("viewCount", 0)),
    }

def get_transcript_text(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except:
        return None

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summary = ""
    for chunk in chunks:
        print(f"Summarizing chunk ({len(chunk)} characters)...")
        result = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
        summary += result[0]['summary_text'] + " "
    return summary.strip()

def get_video_comments(video_id, max_results=100):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet", videoId=video_id,
            maxResults=min(max_results, 100), textFormat="plainText"
        )
        response = request.execute()
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
    except Exception as e:
        print(f"Error fetching comments: {e}")
        return []

    return comments


def analyze_sentiment(comments):
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=device
    )

    label_map = {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "POSITIVE"
    }

    stats = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    results = sentiment_analyzer(comments, truncation=True)

    for res in results:
        label = label_map.get(res["label"], "NEUTRAL")
        stats[label] += 1

    return stats


def plot_sentiment(sentiments):
    labels = list(sentiments.keys())
    sizes = list(sentiments.values())
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Sentiment Distribution of Comments')
    plt.savefig("sentiment_pie_chart.png")
    plt.show()

# ========== Main ==========

def main():
    url = input("Enter YouTube video URL: ")
    video_id = extract_video_id(url)
    
    if not video_id:
        print("Invalid URL!")
        return

    print("\nFetching metadata...")
    metadata = get_video_metadata(video_id)
    print("Title:", metadata["title"])
    print("Channel:", metadata["channel"])
    print("Uploaded on:", metadata["upload_date"])
    print("Views:", metadata["views"])
    print("Likes:", metadata["likes"])

    print("\nFetching transcript...")
    transcript = get_transcript_text(video_id)
    if transcript:
        print("Summarizing transcript...")
        summary = summarize_text(transcript)
        print("\n📄 Video Summary:\n", summary)
    else:
        print("Transcript not available.")

    print("\nFetching comments...")
    comments = get_video_comments(video_id)
    if not comments:
        print("No comments found.")
        return
    
    print("Analyzing sentiment...")
    sentiment_result = analyze_sentiment(comments[:200])  # Limit for speed
    print("\n📊 Sentiment Stats:")
    for k, v in sentiment_result.items():
        print(f"{k}: {v}")

    print("\nPlotting sentiment chart...")
    plot_sentiment(sentiment_result)

if __name__ == "__main__":
    main()
