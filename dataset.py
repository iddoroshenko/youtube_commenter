import pandas as pd
import numpy as np
import pyarrow as pa
import datasets

def load_comments(comments_path, remove_extras, to_lower_case, remove_short_words):
    comments = pd.read_csv(comments_path, error_bad_lines=False)[['video_id', 'comment_text']]
    
    comments.dropna(inplace=True)
    comments.drop(41587, inplace=True)
    comments = comments.reset_index().drop('index',axis=1)
    comments.comment_text = comments.comment_text.astype(str)

    if remove_extras:
        comments['comment_text'] = comments['comment_text'].str.replace("[^a-zA-Z#]", " ")
    if to_lower_case:
        comments['comment_text'] = comments['comment_text'].apply(lambda x:x.lower())
    if remove_short_words:
        comments['comment_text'] = comments['comment_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    
    comments = comments.to_dict('records')
    return comments

def load_videos(videos_path):
    videos_map = pd.read_csv(videos_path, error_bad_lines=False)[['video_id', 'title', 'channel_title']]
    videos_map = dict(zip(videos_map.video_id, zip(videos_map.channel_title, videos_map.title)))
    return videos_map

def load_train_test_dataset(
        comments_path='data/UScomments.csv', 
        videos_path='data/USvideos.csv', 
        split=0.8, 
        remove_extras=False, 
        to_lower_case=False, 
        remove_short_words=False,
        max_len=256,
    ):
    comments_info = load_comments(comments_path, remove_extras, to_lower_case, remove_short_words)
    videos_map = load_videos(videos_path)
    comments_info = [{
        'text': videos_map[c['video_id']][0] + " " + 
                videos_map[c['video_id']][1] +  " " +
                c['comment_text']
        } for c in comments_info
    ]

    comments_info = [c for c in comments_info if len(c) <= max_len]

    np.random.shuffle(comments_info)
    split = int(split * len(comments_info))
    train, test = comments_info[:split], comments_info[split:]

    train = pd.DataFrame(train)
    test = pd.DataFrame(test)

    train = datasets.Dataset(pa.Table.from_pandas(train))
    test = datasets.Dataset(pa.Table.from_pandas(test))
    return train, test