# unsupervised learning
# when we have unlabelled data 
# there is no for sure answer to such questions
# LDA uses the dirichlet probability distribution
# it gives distribution of topics to which a set of text articles belong to and in those topics it gives the distribution of the words
import pandas as pd
npr = pd.read_csv('D:\\NLP\\UPDATED_NLP_COURSE\\UPDATED_NLP_COURSE\\05-Topic-Modeling\\npr.csv')
print(npr.head())
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = tfidf.fit_transform(npr['Article'])
from sklearn.decomposition import NMF
nmf_model = NMF(n_components=7,random_state=42)
# This can take awhile, we're dealing with a large amount of documents!
nmf_model.fit(dtm)
import random
for i in range(10):
    random_word_id = random.randint(0,54776)
    print(tfidf.get_feature_names_out()[random_word_id])
for i in range(10):
    random_word_id = random.randint(0,54776)
    print(tfidf.get_feature_names_out()[random_word_id])

single_topic = nmf_model.components_[0]
top_word_indices = single_topic.argsort()[-10:]
topic_results = nmf_model.transform(dtm)
npr['Topic'] = topic_results.argmax(axis=1)
print(npr.head(10))
print('done')
# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer(max_df=0.95,min_df=2,stop_words='english')
# dtm = cv.fit_transform(npr['Article'])

# from sklearn.decomposition import LatentDirichletAllocation
# LDA = LatentDirichletAllocation(n_components=7,random_state=42)
# LDA.fit(dtm)

