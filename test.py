import uvicorn
from fastapi import FastAPI, Body, Request, File, UploadFile, Form
from fastapi import responses
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pdfplumber
import re
import aiofiles
import pickle


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

accuracy = 98.0


#LOAD MODELS
label_file = 'finalized_label2.sav'
tfidf_file = 'finalized_tfidf2.sav'
clf_file = 'finalized_clf2.sav'

le = pickle.load(open(label_file, 'rb'))
cv = pickle.load(open(tfidf_file, 'rb'))
clf = pickle.load(open(clf_file, 'rb'))
print("Yoo done")


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="htmlviews")


@app.get("/")
def home(request:Request):
    return templates.TemplateResponse("index.html",{"request":request,"accuracy":accuracy})


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

def solve(filepath):
    resumeText = extractText(filepath)
    cleanedText = cleanResume(resumeText)
    text = re.split("curricular",cleanedText)
    textp = cv.transform([text[0]]).toarray()
    prediction2 = clf.predict(textp)
    print(prediction2[0]);
    predictedResult = le.inverse_transform(prediction2)[0]    
    return predictedResult

def extractText(filepath):
    with pdfplumber.open(filepath) as pdf:
        first_page = pdf.pages[0]
        resumeText = first_page.extract_text()        
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
