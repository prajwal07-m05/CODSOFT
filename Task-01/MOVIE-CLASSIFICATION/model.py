import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
def read_train_file(path):
    ids=[]
    titles=[]
    genres=[]
    desc=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            parts=line.strip().split(":::")
            if len(parts)>=4:
                ids.append(parts[0].strip())
                titles.append(parts[1].strip())
                genres.append(parts[2].strip())
                desc.append(parts[3].strip())
    data=pd.DataFrame({"id":ids,"title":titles,"genre":genres,"description":desc})
    return data
def read_test_file(path):
    ids=[]
    titles=[]
    desc=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            parts=line.strip().split(":::")
            if len(parts)>=3:
                ids.append(parts[0].strip())
                titles.append(parts[1].strip())
                desc.append(parts[2].strip())
    data=pd.DataFrame({"id":ids,"title":titles,"description":desc})
    return data
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-z\s]','',text)
    return text
print("Reading training dataset...")
train_df=read_train_file("train_data.txt")
print("Training samples:",len(train_df))
train_df["description"]=train_df["description"].apply(clean_text)
print("Converting text into TF-IDF features...")
vectorizer=TfidfVectorizer(stop_words="english",max_features=5000)
X_train=vectorizer.fit_transform(train_df["description"])
y_train=train_df["genre"]
print("Training Naive Bayes model...")
model=MultinomialNB()
model.fit(X_train,y_train)
print("Reading test dataset...")
test_df=read_test_file("test_data.txt")
test_df["description"]=test_df["description"].apply(clean_text)
X_test=vectorizer.transform(test_df["description"])
print("Predicting genres...")
predictions=model.predict(X_test)
test_df["predicted_genre"]=predictions
print("Sample predictions:")
print(test_df[["title","predicted_genre"]].head())
solution=read_train_file("test_data_solution.txt")
accuracy=accuracy_score(solution["genre"],predictions)
print("Model Accuracy:",accuracy)