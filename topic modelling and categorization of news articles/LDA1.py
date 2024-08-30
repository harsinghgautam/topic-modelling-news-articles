# unsupervised learning
# when we have unlabelled data 
# there is no for sure answer to such questions
# LDA uses the dirichlet probability distribution
# it gives distribution of topics to which a set of text articles belong to and in those topics it gives the distribution of the words
import pandas as pd
npr = pd.read_csv('D:\\NLP\\UPDATED_NLP_COURSE\\UPDATED_NLP_COURSE\\05-Topic-Modeling\\npr.csv')
print(npr.head())

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df=0.95,min_df=2,stop_words='english')
dtm = cv.fit_transform(npr['Article'])

from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=7,random_state=42)
LDA.fit(dtm)

print(len(cv))