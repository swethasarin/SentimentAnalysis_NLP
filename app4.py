# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

nltk.download("stopwords")
nltk.download("wordnet")

# ================================
# Load Data
# ================================
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\HP\Downloads\Datascience_5th_project\chatgpt_reviews.csv")

df = load_data()

# ================================
# Preprocessing
# ================================
stop_words = set(stopwords.words("english")) - {"not", "no", "good", "bad", "great", "best"}
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(tokens)

df["cleaned_review"] = df["review"].astype(str).apply(clean_text)

# ================================
# Sentiment Labels
# ================================
def label_sentiment(r):
    if r <= 2:
        return "Negative"
    elif r == 3:
        return "Neutral"
    else:
        return "Positive"

df["sentiment"] = df["rating"].apply(label_sentiment)

# ================================
# Balance Dataset for Model
# ================================
df_majority = df[df.sentiment == "Negative"]
df_minority_pos = df[df.sentiment == "Positive"]
df_minority_neu = df[df.sentiment == "Neutral"]

df_minority_pos_up = resample(df_minority_pos, replace=True, n_samples=len(df_majority), random_state=42)
df_minority_neu_up = resample(df_minority_neu, replace=True, n_samples=len(df_majority), random_state=42)

df_balanced = pd.concat([df_majority, df_minority_pos_up, df_minority_neu_up])

X_train, X_test, y_train, y_test = train_test_split(
    df_balanced["cleaned_review"], df_balanced["sentiment"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=500, class_weight="balanced")
model.fit(X_train_vec, y_train)

# ================================
# Streamlit Layout
# ================================
st.title("AI Echo: Sentiment Analysis of ChatGPT Reviews")

option = st.selectbox(
    "📊 Select a visualization:",
    [
        "1. Distribution of review ratings",
        "2. Helpful reviews count",
        "3. Positive vs Negative keywords",
        "4. Average rating over time",
        "5. Ratings by location",
        "6. Platform comparison",
        "7. Verified vs Non-verified",
        "8. Review length vs Rating",
        "9. Top words in 1-star reviews",
        "10. Ratings by ChatGPT version",
        "11. Sentiment distribution",
    ],
)

# ================================
# Visualizations
# ================================
if option.startswith("1"):
    st.subheader("Distribution of Ratings")
    fig, ax = plt.subplots()
    df["rating"].value_counts().sort_index().plot(kind="bar", ax=ax, color="skyblue")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    st.pyplot(fig)

elif option.startswith("2"):
    st.subheader("Helpful Reviews (Top 20)")
    top_helpful = df.sort_values("helpful_votes", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.barh(top_helpful["title"], top_helpful["helpful_votes"], color="orange")
    ax.set_xlabel("Helpful Votes")
    ax.set_ylabel("Review Title")
    st.pyplot(fig)

elif option.startswith("3"):
    st.subheader("Word Clouds for Positive vs Negative Reviews")
    pos_text = " ".join(df[df["sentiment"] == "Positive"]["cleaned_review"])
    neg_text = " ".join(df[df["sentiment"] == "Negative"]["cleaned_review"])
    wc_pos = WordCloud(width=400, height=200, background_color="white").generate(pos_text)
    wc_neg = WordCloud(width=400, height=200, background_color="black", colormap="Reds").generate(neg_text)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(wc_pos); ax1.axis("off"); ax1.set_title("Positive")
    ax2.imshow(wc_neg); ax2.axis("off"); ax2.set_title("Negative")
    st.pyplot(fig)

elif option.startswith("4"):
    st.subheader("Average Rating Over Time")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    avg_time = df.groupby(df["date"].dt.to_period("M"))["rating"].mean()
    fig, ax = plt.subplots()
    avg_time.plot(ax=ax, marker="o")
    ax.set_ylabel("Average Rating")
    st.pyplot(fig)

elif option.startswith("5"):
    st.subheader("Ratings by Location (Top 10)")
    top_loc = df["location"].value_counts().head(10).index
    loc_data = df[df["location"].isin(top_loc)]
    fig, ax = plt.subplots(figsize=(10,5))
    loc_data.groupby("location")["rating"].mean().sort_values().plot(kind="bar", ax=ax, color="green")
    ax.set_ylabel("Average Rating")
    st.pyplot(fig)

elif option.startswith("6"):
    st.subheader("Platform Comparison (Average Rating)")
    fig, ax = plt.subplots()
    df.groupby("platform")["rating"].mean().plot(kind="bar", ax=ax, color="purple")
    ax.set_ylabel("Average Rating")
    st.pyplot(fig)

elif option.startswith("7"):
    st.subheader("Verified vs Non-verified")
    fig, ax = plt.subplots()
    df.groupby("verified_purchase")["rating"].mean().plot(kind="bar", ax=ax, color="teal")
    ax.set_ylabel("Average Rating")
    st.pyplot(fig)

elif option.startswith("8"):
    st.subheader("Review Length vs Rating")
    fig, ax = plt.subplots()
    df.groupby("rating")["review_length"].mean().plot(kind="bar", ax=ax, color="pink")
    ax.set_ylabel("Avg Review Length")
    st.pyplot(fig)

elif option.startswith("9"):
    st.subheader("Top Words in 1-star Reviews")
    text_1star = " ".join(df[df["rating"] == 1]["cleaned_review"])
    wc = WordCloud(width=600, height=400, background_color="white").generate(text_1star)
    fig, ax = plt.subplots()
    ax.imshow(wc); ax.axis("off")
    st.pyplot(fig)

elif option.startswith("10"):
    st.subheader("Ratings by ChatGPT Version")
    fig, ax = plt.subplots()
    df.groupby("version")["rating"].mean().plot(kind="bar", ax=ax, color="brown")
    ax.set_ylabel("Average Rating")
    st.pyplot(fig)

elif option.startswith("11"):
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    df["sentiment"].value_counts().plot(kind="bar", ax=ax, color="cyan")
    st.pyplot(fig)

# ================================
# Prediction Section
# ================================
st.subheader("🔮 Try Your Own Review")
user_input = st.text_area("Enter a review:")
if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        st.write(f"### Predicted Sentiment: {prediction}")
    else:
        st.warning("Please enter some text!")
