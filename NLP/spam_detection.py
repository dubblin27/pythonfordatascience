import nltk 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC




# nltk.download_shell() 

#spam or ham 

# messages = [line.rstrip() for line in open('NLP/SMSSpamCollection')]
# print(len(messages))
# for mess_no,message in enumerate(messages[:10]):
#     print(mess_no , message) 

messages = pd.read_csv('NLP/SMSSpamCollection',sep = '\t',names = ['label','message'])
# print(messages.head(), messages.describe()) 

# print(messages.groupby('label').describe()) 
messages['length'] = messages['message'].apply(len)
# print(messages.head())

messages['length'].plot.hist(bins=60)
# plt.show() 

# print(messages['length'].describe()) 

# longest message 
# print(messages[messages['length'] == 910]['message'].iloc[0])
#ham vs spam visualization
messages.hist(column = 'length', by='label', bins=60,figsize=(12,4))
# plt.show()
# nopunc = ''.join(nopunc)

# to remove shitty words 
def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# print(messages['message'].head().apply(text_process))

#vectorization 

#to count the occurance - frequency
#weigh the counts - IDF 
# then normalize 

bow_transformer = CountVectorizer(analyzer= text_process).fit(messages['message'])
# print(len(bow_transformer.vocabulary_))

mess4 = messages['message'][3:10]
# print(mess4)

bow4 = bow_transformer.transform([mess4])
# print(bow4.shape)
# print(bow4)


message_bow = bow_transformer.transform(messages['message'])
# print(message_bow.shape)
#non zeros
# print(message_bow.nnz)
# total non zeros to all the messages

sparsity = (100.0 * message_bow.nnz / (message_bow.shape[0] * message_bow.shape[1]))
# print('sparsity: {}'.format(round(sparsity)))

tfidf_transformer = TfidfTransformer().fit(message_bow)
tfidf4 = tfidf_transformer.transform(bow4)
# print(tfidf4)
# to view the occurances of the words
# print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

messages_tfidf = tfidf_transformer.transform(message_bow) 

spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])
all_pred = spam_detect_model.predict(messages_tfidf)
# print(all_pred)

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'],messages['label'],test_size = 0.3)
# print(msg_train) 

pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline1 = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier', KNeighborsClassifier())
])
pipeline2 = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier', DecisionTreeClassifier())
])
pipeline3 = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier', RandomForestClassifier())
])
pipeline4 = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier', SVC())
])

pipeline.fit(msg_train,label_train)

predictions = pipeline.predict(msg_test)
# print(predictions)
print("using Naive Bayes : ")
print(classification_report(label_test,predictions))

print("using KNN ")
pipeline1.fit(msg_train,label_train)

predictions1 = pipeline1.predict(msg_test)
print(classification_report(label_test,predictions1))

print("using Decision Tree:  ")
pipeline2.fit(msg_train,label_train)

predictions2 = pipeline2.predict(msg_test)
print(classification_report(label_test,predictions2))

print("using Random Forest: ")
pipeline3.fit(msg_train,label_train)

predictions3 = pipeline3.predict(msg_test)
print(classification_report(label_test,predictions3))

print("using SVM: ")
pipeline4.fit(msg_train,label_train)

predictions4 = pipeline4.predict(msg_test)
print(classification_report(label_test,predictions4))
