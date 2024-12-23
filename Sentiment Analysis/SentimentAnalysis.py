import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

sentiment_df = pd.read_csv('/mnt/data/helpdesk_customer_multi_lang_tickets.csv')

def preprocess_text(df):
    df = df[['body', 'priority']].dropna()
    df['priority'] = df['priority'].map({'High': 1, 'Medium': 0, 'Low': 0})  # Binary sentiment
    return df

sentiment_df = preprocess_text(sentiment_df)

X_train, X_test, y_train, y_test = train_test_split(
    sentiment_df['body'], sentiment_df['priority'], test_size=0.2, random_state=42
)

vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred) * 100

with open('sentiment_accuracy.txt', 'w') as f:
    f.write(f"Sentiment Analysis Accuracy: {accuracy:.2f}%")

print(f"Sentiment Analysis completed with {accuracy:.2f}% accuracy.")
