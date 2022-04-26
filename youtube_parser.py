import urllib
import re
from googleapiclient.discovery import build

def get_video_id_by_url(url):
    parsed_url = urllib.parse.urlparse(url)
    video_id = urllib.parse.parse_qs(parsed_url.query).get("v")
    return video_id[0]

def youtube_authenticate():
    with open('api_key.txt') as f:
        api_key = f.readlines()[0]
    return build("youtube", "v3", developerKey=api_key)

def get_video_details(youtube, **kwargs):
    return youtube.videos().list(
        part="snippet,contentDetails,statistics",
        **kwargs
    ).execute()

def parse_response(video_response):
    items = video_response.get("items")[0]
    snippet         = items["snippet"]
    channel_title = snippet["channelTitle"]
    title         = snippet["title"]
    description   = snippet["description"]
    return {
        "Title": re.sub(' +', ' ', title),
        "Description": description,
        "Channel": re.sub(' +', ' ',channel_title),
    }

def get_info_from_url(url):
    video_id = get_video_id_by_url(url)
    youtube = youtube_authenticate()
    response = get_video_details(youtube, id=video_id)
    return parse_response(response)

