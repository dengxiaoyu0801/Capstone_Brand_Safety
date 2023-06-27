import praw # module which makes it easier to interact with api
import pandas as pd
from functools import lru_cache

def init_praw():
    """
        return praw object used to make api calls
    """
    reddit = praw.Reddit(
        client_id="j0O0m4-YDq4Y2_W9GMv2Pg",
        client_secret="B2wKh_UpgcZdDiurxqGnGIOl6Z6umw",
        user_agent='bens_api_app/0.0.1',
    )

    return reddit

@lru_cache()
def fetch_and_format_psts(reddit, username):
    """
        Args:
            reddit: the praw instance from init praw
            username: string, the username
           
        returns:
            pandas data frame with index, 1 data column with the
                string from the post
            pandas dataframe with index, 1 data column with text
                from the comment
       
    """
    posts = list(reddit.redditor(username).submissions.new(limit=None))
    comments = list(reddit.redditor(username).comments.new(limit=None))

    post_text = [
            post.selftext for post in posts if (
                post.selftext != '[removed]' and post.selftext != ''
                )
        ]
   
    comment_text = [
        comment.body for comment in comments if (
            comment.body != '[removed]' and comment.body != ''
            )
    ]

    df_text = pd.DataFrame(post_text+comment_text,columns =['text'])

    return df_text