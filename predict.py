
import joblib
#import api as api
import api_reddit as redditapi
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
import pandas as pd

def vect(test_X):
    def _fetch_train_data():
        pdfs = []
        for i in [1,2,3,4]:
            pdfs.append(pd.read_csv(f'train_part_{i}.csv')
        return pd.concat(pdfs)
                       
    vect = TfidfVectorizer(max_features=5000,stop_words='english')
    train_df = _fetch_train_data()
    X = train_df['comment_text']
    X_dtm = vect.fit_transform(X)
    test_X_dtm = vect.transform(test_X)
    return test_X_dtm

def add_feature(X, feature_to_add):
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

def main_predict(userid):

    #tw_df = api.format_api_output(userid)
    reddit = redditapi.init_praw()
    reddit_df = redditapi.fetch_and_format_psts(reddit, userid)
    cols_target = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']
    test_X = reddit_df['text']
    test_X_dtm = vect(test_X)



    filename_binary = "Completed_model_binary.joblib"
    logreg_binary = joblib.load(filename_binary)
    submission_binary = pd.DataFrame()

    for label in cols_target:
        y_pred_X = logreg_binary.predict(test_X_dtm)
        #print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))
        # compute the predicted probabilities for X_test_dtm
        test_y_prob = logreg_binary.predict_proba(test_X_dtm)[:,1]
        submission_binary[label] = test_y_prob


    submission_chains = pd.DataFrame()
    for index,label in enumerate(cols_target):

        num = '{}'.format(index+1)
        filename_chain = 'Completed_model_chain{}_{}.joblib'.format(num,label)
        logreg_chain = joblib.load(filename_chain)
        test_y = logreg_chain.predict(test_X_dtm)

        test_y_prob = logreg_chain.predict_proba(test_X_dtm)[:,1]
        submission_chains[label] = test_y_prob

        test_X_dtm = add_feature(test_X_dtm, test_y)

    submission_combined = pd.DataFrame() 
    submission = pd.DataFrame()
    for label in cols_target:
        submission_combined[label] = 0.5*(submission_chains[label]+submission_binary[label])
        perc = [submission_combined[submission_combined[label]>0.3][label].count()
                /submission_combined[label].count()]

        submission[label] = perc
    df_pred = pd.concat([reddit_df,submission_combined.round(2)],axis = 1)
    submission.iloc[:, 0:] = submission.iloc[:, 0:].round(4)*100
    submission = submission.rename(index={0: 'percentage'})
    df_filter = df_pred[(df_pred.iloc[:,1:] > 0.3).any(1)]
    score = '{:.1%}'.format(round(df_filter.shape[0]/df_pred.shape[0], 4))
    return df_pred,df_filter,submission,score
    
    
    
    
    
