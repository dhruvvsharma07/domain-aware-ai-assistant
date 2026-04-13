import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#PHASE 1: DATA PREPARATION
# =========================
# Load Dataset
# =========================
with open("data.json", "r") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

# =========================
# Retriever
# =========================
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(questions)

def get_best_match(user_query):
    query_vec = vectorizer.transform([user_query])
    similarity = cosine_similarity(query_vec, X)
    best_idx = similarity.argmax()
    score = similarity[0][best_idx]
    return best_idx, score

def chatbot_response(user_query):
    idx, score = get_best_match(user_query)

    if score > 0.3:
        return answers[idx], score
    else:
        return "I couldn't find an exact answer. Try rephrasing.", score

# =========================
# UI
# =========================
st.set_page_config(page_title="UNIT 5 Syllabus Assistant")

st.title("UNIT 5 Syllabus Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask something REGARDING UNIT 5:")

if user_input:
    response, score = chatbot_response(user_input)

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Teacher", response))

# Display chat
for role, text in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {text}")
    else:
        st.markdown(f"**👨‍🏫 Teacher:** {text}")

#PHASE 2 
