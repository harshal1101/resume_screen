import uvicorn
# from typing import Optional
from fastapi import FastAPI, Body, Request, File, UploadFile, Form
from fastapi import responses
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier

import seaborn as sns
import pdfplumber
import re
import aiofiles
 


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




df = pd.read_csv('UpdatedResumeDataSet.csv',encoding = 'utf-8')
#print(df.head().encode('utf-8'))

#print(df['Category'].unique())

#print(df['Category'].value_counts())

plt.figure(figsize=(20,20))
sns.countplot(y='Category',data=df)
df['modified_resume'] =''
df['modified_resume'] = df['Resume'].apply(lambda x: cleanResume(x))


import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import string

stopwordsList = set(stopwords.words('english')+['``',"''"])
sentences = df['Resume'].values   #returning only the values of the column without any axis label
cleanedSentences = ""
allwords =[]
for i in range(len(sentences)):
    cleanedText = cleanResume(sentences[i])        
    cleanedSentences += cleanedText
    words = nltk.word_tokenize(cleanedText)
    for word in words:
        if word not in stopwordsList and word not in string.punctuation:
            allwords.append(word)

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

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
catChange = ['Category']
for i in catChange:   
    df[i] = le.fit_transform(df[i])
#catChange


from sklearn.feature_extraction.text import TfidfVectorizer
requiredText = df['modified_resume'].values
cv = TfidfVectorizer(stop_words='english',max_features = 500)

WordFeatures = cv.fit_transform(requiredText).toarray()



from sklearn.model_selection import train_test_split

requiredTarget = df['Category'].values
X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=0, test_size=0.2)
print("yesss")
print(X_train.shape)
print(X_test.shape)


clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
clf.score(X_test, y_test)
print(prediction)

#mlModel()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="htmlviews")


@app.get("/")
def home(request:Request):
    return templates.TemplateResponse("index.html",{"request":request})


@app.post("/submitfile",response_class=HTMLResponse)
async def func(request: Request,uploadedFile: UploadFile = File(...)):    
    print(uploadedFile.content_type)
    content_file = await uploadedFile.read()
    filepath = "docs/" + uploadedFile.filename.replace(" ","-")
    with open(filepath, 'wb') as f:        
        f.write(content_file)
        f.close()
    ans = solve(filepath)
    return templates.TemplateResponse("predict.html",{"request":request,"ans":ans})
    # return{
    #     "ans": ans
    # }
    

    #print(content_file)
def solve(filepath):
    resumeText = extractText(filepath)
    cleanedText = cleanResume(resumeText)
    text = re.split("curricular",cleanedText)
    textp = cv.transform([text[0]]).toarray()
    prediction2 = clf.predict(textp)
    print(prediction2[0]);
    predictedResult = le.inverse_transform(prediction2)[0]
    #return prediction2[0].item()   numpy.int32 to int
    return predictedResult

def extractText(filepath):
    with pdfplumber.open(filepath) as pdf:
        first_page = pdf.pages[0]
        resumeText = first_page.extract_text()
        #print(resumeText)
        pattern = r'[0-9]'
        # # Match all digits in the string and replace them with an empty string
        resumeText= re.sub(pattern, '', resumeText)
        return resumeText



# pattern = r'[0-9]'

# # Match all digits in the string and replace them with an empty string
# checkstring= re.sub(pattern, '', checkstring)
# checkstring = cleanResume(checkstring)
# print(checkstring)


#app = FastAPI()

#lass stringValue(BaseModel):
   #stringg:str

#app.get('/')
#ef yo():
   #return{"ms":"hello"}
#app.post("/submit")
#ef handle_form(string_value: stringValue):
   #print(string_value)
    #eturn{"yp":"yess"}
    
    

# if __name__ == '__main__':
#   uvicorn.run(app, host='localhost', port=3050)