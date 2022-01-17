import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import re
import pickle



df = pd.read_csv('UpdatedResumeDataSet.csv',encoding = 'utf-8')

#FUNCTION FOR CLEANING THE RESUME TEXT
def cleanResume(resumeText):
    resumeText = resumeText.lower()
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

plt.figure(figsize=(20,20))
sns.countplot(y='Category',data=df)
df['modified_resume'] =''
df['modified_resume'] = df['Resume'].apply(lambda x: cleanResume(x))


#ENCODING THE TARGET CATEGORICAL VALUES

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cat_values = df['Category'].values
df['Category'] = le.fit_transform(cat_values)
# catChange = ['Category']
# for i in catChange:   
#     df[i] = le.fit_transform(df[i])


# EXTRACTING FEATURES FROM THE RESUME TEXT
from sklearn.feature_extraction.text import TfidfVectorizer
requiredText = df['modified_resume'].values
cv = TfidfVectorizer(stop_words='english',max_features = 500)

WordFeatures = cv.fit_transform(requiredText).toarray()


# TRAINING AND TESTING DATA
from sklearn.model_selection import train_test_split

requiredTarget = df['Category'].values
X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=0, test_size=0.2)
print("yesss")
print(X_train.shape)
print(X_test.shape)



# CLASSIFIER
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print(clf.score(X_test, y_test))
print(prediction)
accuracy = round(accuracy_score(y_test, prediction),2)
f_score = f1_score(y_test, prediction, average='weighted')
print("Accuracy is " + str(accuracy))
print("Weighted-average f1-score is " + str(f_score))
print("Model working")



# SAVING MODELS
label_file = 'finalized_label2.sav'
pickle.dump(le, open(label_file, 'wb'))

tfidf_file = 'finalized_tfidf2.sav'
pickle.dump(cv, open(tfidf_file, 'wb'))

clf_file = 'finalized_clf2.sav'
pickle.dump(clf, open(clf_file, 'wb'))





# WORDCLOUD IMAGE

# import nltk
# from nltk.corpus import stopwords
# from wordcloud import WordCloud
# import string

# stopwordsList = set(stopwords.words('english')+['``',"''"])
# sentences = df['Resume'].values   #returning only the values of the column without any axis label
# cleanedSentences = ""
# allwords =[]
# for i in range(len(sentences)):
#     cleanedText = cleanResume(sentences[i])        
#     cleanedSentences += cleanedText
#     words = nltk.word_tokenize(cleanedText)
#     for word in words:
#         if word not in stopwordsList and word not in string.punctuation:
#             allwords.append(word)

# wordFreq = nltk.FreqDist(allwords)
# mostFreq = wordFreq.most_common(30)
# print(mostFreq)
# wordCloud = WordCloud(width=1920, height=1080,max_font_size = 256,background_color='#6dd5ed',colormap="gist_heat").generate(cleanedSentences)

# plt.figure(figsize=(10,10))
# plt.imshow(wordCloud)
# plt.axis("off")
# plt.show()

# from PIL import Image
# mask = np.array(Image.open('WhatsApp Image 2021-06-27 at 19.58.09.jpeg'))
# wordCloud = WordCloud(max_font_size = 256,background_color='white',mask=mask).generate(cleanedSentences)
# plt.figure(figsize=(10,10))
# plt.imshow(wordCloud)
# plt.axis("off")
# plt.show()

#df['Category'].unique()