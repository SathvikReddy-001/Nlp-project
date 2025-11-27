import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Indian Law Summariser",
    layout="centered",
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.main { background-color: #000000; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* REMOVE BLANK INPUT SHADOW/BG */
.stTextInput > div > div {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.law-card {
    background-color: #111111;
    padding: 1.2rem 1.5rem;
    border-radius: 0.9rem;
    border: 1px solid #222222;
    margin-top: 1rem;
}

.law-pill {
    display: inline-block;
    padding: 0.15rem 0.7rem;
    border-radius: 999px;
    background-color: #222222;
    font-size: 0.8rem;
    color: #cccccc;
    margin-bottom: 0.5rem;
}

.law-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown("<h1 style='text-align:center; color:white;'>Indian Law Summariser</h1>", unsafe_allow_html=True)
st.write("Type an Article (Article 14, Article 21, Article 300A) or a legal term (Lease Deed, Adoption Deed) to see a simple explanation.")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_clean_law_dataset.csv")
    df["combined"] = df["section"].astype(str) + " " + df["text"].astype(str)
    return df

df = load_data()

# ------------------ TF-IDF MODEL ------------------
@st.cache_resource
def build_model(dataframe):
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform(dataframe["combined"])
    return tfidf, vectors

tfidf, vectors = build_model(df)

# ------------------ CATEGORY DETECTOR ------------------
def detect_category(section):
    return "Constitution Article" if section.lower().startswith("article") else "Legal Document / Concept"

# ------------------ SMART SEARCH ENGINE ------------------
def search_law(query):
    query_low = query.lower().strip()

    # 1️⃣ Exact match
    exact_match = df[df["section"].str.lower() == query_low]
    if len(exact_match) > 0:
        return exact_match.iloc[0]

    # 2️⃣ Number-only match
    number_match = df[df["section"].str.lower().str.contains(rf"\b{query_low}\b")]
    if len(number_match) == 1:
        return number_match.iloc[0]

    # 3️⃣ TF-IDF fallback
    q_vec = tfidf.transform([query])
    similarity = cosine_similarity(q_vec, vectors).flatten()
    idx = similarity.argmax()
    return df.iloc[idx]

# ------------------ USER INPUT ------------------
user_input = st.text_input(
    "Search:",
    placeholder="Example: Article 22, Article 300A, right to equality, Lease Deed"
)

# ------------------ RESULT DISPLAY ------------------
if user_input:
    st.markdown("### Result")
    
    result = search_law(user_input)
    category = detect_category(result["section"])

    # START CARD
    st.markdown("<div class='law-card'>", unsafe_allow_html=True)

    # Category pill
    st.markdown(f"<div class='law-pill'>{category}</div>", unsafe_allow_html=True)

    # Title
    st.markdown(f"<div class='law-title'>{result['section']}</div>", unsafe_allow_html=True)

    # Simple meaning
    st.markdown("**Meaning in Simple Terms**")
    st.write(result["simple_meaning"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Original legal text
    st.markdown("**Original Legal Text**")
    st.write(result["text"])

    # END CARD
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Enter an article number or legal term above.")
