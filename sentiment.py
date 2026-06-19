from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re
import pandas as pd
columns = ["sentiment", "id", "date", "query", "user", "text"]
data = pd.read_csv(
    "data/training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    header=None,
    names=columns
)
data["sentiment"] = data["sentiment"].replace(4, 1)
# Take only 10000 rows
data = data.sample(10000, random_state=42)
print(data.shape)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text
data["clean_text"] = data["text"].apply(clean_text)
print(data[["text", "clean_text"]].head())
X = data["clean_text"]
y = data["sentiment"]
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
user_text = input("Enter a sentence: ")
user_text = clean_text(user_text)
user_vector = vectorizer.transform([user_text])
prediction = model.predict(user_vector)
if prediction[0] == 1:
    print("Sentiment: Positive 😊")
else:
    print("Sentiment: Negative 😞")