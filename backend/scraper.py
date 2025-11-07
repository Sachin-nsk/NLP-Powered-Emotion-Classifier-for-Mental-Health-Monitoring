# scraper.py (Final, Robust Version)

import instaloader
import praw
import tweepy
import os
import re
from dotenv import load_dotenv

# ----------------- CONFIGURATION & INITIALIZATION -----------------

load_dotenv()

# --- Instagram Setup ---
# (No changes needed here, instaloader is already quite robust)
INSTA_USERNAME = os.getenv("INSTA_USERNAME")
INSTA_PASSWORD = os.getenv("INSTA_PASSWORD")
L = instaloader.Instaloader()
try:
    print("Attempting to log in to Instagram...")
    L.load_session_from_file(INSTA_USERNAME)
    print("Instagram session loaded successfully.")
except FileNotFoundError:
    print("No session file found. Logging in with credentials...")
    L.login(INSTA_USERNAME, INSTA_PASSWORD)
    L.save_session_to_file(INSTA_USERNAME)
    print("Instagram login successful and session saved.")

# --- Reddit Setup ---
# (No code change here, just ensure your .env file is correct)
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)
print("Reddit client initialized.")

# --- Twitter/X Setup ---
# UPDATED: Added wait_on_rate_limit=True to handle 429 errors
x_client = tweepy.Client(os.getenv("X_BEARER_TOKEN"), wait_on_rate_limit=True)
print("Twitter/X client initialized.")


# ----------------- PLATFORM-SPECIFIC SCRAPERS -----------------
# (No changes needed in the functions themselves)

def get_shortcode_from_url(post_url: str):
    match = re.search(r"/(p|reel)/([^/]+)", post_url)
    if match: return match.group(2)
    return None

def fetch_instagram_post_data(post_url: str):
    print(f"Fetching Instagram post: {post_url}")
    shortcode = get_shortcode_from_url(post_url)
    if not shortcode: raise ValueError("Invalid Instagram post URL provided.")
    try:
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        caption = post.caption if post.caption else "No caption found."
        media_url = post.url
        return {"caption": caption, "media_urls": [media_url]}
    except Exception as e:
        raise RuntimeError(f"Failed to fetch Instagram post {shortcode}: {e}")

def fetch_reddit_post_data(url: str):
    print(f"Fetching Reddit post: {url}")
    submission = reddit.submission(url=url)
    caption = submission.title
    if submission.selftext: caption += f" | {submission.selftext}"
    media_url = None
    if not submission.is_self and hasattr(submission, 'url') and submission.url.endswith(('.jpg', '.jpeg', '.png')):
        media_url = submission.url
    return {"caption": caption, "media_urls": [media_url] if media_url else []}

def fetch_twitter_post_data(url: str):
    print(f"Fetching Tweet: {url}")
    tweet_id = url.split("/")[-1].split("?")[0]
    response = x_client.get_tweet(
        tweet_id, 
        expansions=["attachments.media_keys"], 
        media_fields=["url", "preview_image_url"]
    )
    caption = response.data.text
    media_url = None
    if response.includes and "media" in response.includes:
        for media_item in response.includes["media"]:
            if media_item.url:
                media_url = media_item.url; break
            elif media_item.preview_image_url:
                media_url = media_item.preview_image_url; break
    return {"caption": caption, "media_urls": [media_url] if media_url else []}


# ----------------- MASTER FUNCTION -----------------

def fetch_post_data_from_any_url(url: str):
    if "instagram.com" in url:
        return fetch_instagram_post_data(url)
    elif "reddit.com" in url:
        return fetch_reddit_post_data(url)
    elif "twitter.com" in url or "x.com" in url:
        return fetch_twitter_post_data(url)
    else:
        raise ValueError("Unsupported URL. Please provide a valid Instagram, Reddit, or Twitter/X URL.")