import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
import os
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# API Keys
groq_api_key = os.getenv("GROQ_API_KEY")

# Langsmith Tracking (optional)

#if st.secrets.get("LANGCHAIN_API_KEY"):
#    os.environ['LANGCHAIN_API_KEY'] = st.secrets["LANGCHAIN_API_KEY"]
#    os.environ["LANGCHAIN_TRACING_V2"] = "true"
#    os.environ["LANGCHAIN_PROJECT"] = "Sentiment Analysis ChatBot"


# LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.1-8b-instant"
)

# Streamlit UI Configuration
st.set_page_config(
    page_title="Sentiment Analysis Chatbot",
    page_icon="😊",
    layout="wide"
)

# Title and Description
st.title("😊 Sentiment Analysis Chatbot")
st.markdown("Share your thoughts, feelings, or any text and get detailed sentiment analysis with AI-powered insights!")

# Sidebar Configuration
st.sidebar.header("⚙️ Settings")
username = st.sidebar.text_input("Enter your username", value="user").replace(" ", "")
session_id = st.sidebar.text_input("Session ID", value=f"{username}_sentiment_session")

# Initialize session state
if "store" not in st.session_state:
    st.session_state.store = {}

if "sentiment_history" not in st.session_state:
    st.session_state.sentiment_history = []

# Sidebar - Analysis Options
st.sidebar.header("📊 Analysis Options")
show_detailed_analysis = st.sidebar.checkbox("Show Detailed Analysis", value=True)
show_emotion_breakdown = st.sidebar.checkbox("Show Emotion Breakdown", value=True)
show_sentiment_trend = st.sidebar.checkbox("Show Sentiment Trend", value=True)

# Helper Functions
def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Classify sentiment
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return {
        "sentiment": sentiment,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "confidence": abs(polarity)
    }

def get_emotion_from_sentiment(polarity, subjectivity):
    """Map sentiment scores to emotions"""
    if polarity >= 0.5:
        return "Joy" if subjectivity > 0.5 else "Content"
    elif polarity >= 0.1:
        return "Hope" if subjectivity > 0.5 else "Calm"
    elif polarity <= -0.5:
        return "Anger" if subjectivity > 0.5 else "Sadness"
    elif polarity <= -0.1:
        return "Worry" if subjectivity > 0.5 else "Disappointment"
    else:
        return "Neutral"

# Create sentiment analysis chain
sentiment_system_prompt = """
You are an expert sentiment analysis assistant with deep understanding of human emotions and psychology. 

Your task is to:
1. Analyze the sentiment and emotional tone of the user's text
2. Provide insights into the underlying emotions and feelings
3. Offer supportive and empathetic responses
4. Give constructive feedback or suggestions when appropriate
5. Help users understand their emotional state better

Guidelines:
- Be empathetic and supportive in your responses
- Provide specific insights about the sentiment and emotions detected
- Offer helpful suggestions or perspectives when appropriate
- Keep responses conversational and engaging
- If the sentiment is negative, provide gentle support and encouragement
- If the sentiment is positive, acknowledge and celebrate it
- Always be respectful and non-judgmental

The sentiment analysis data will be provided to you along with the user's message.
"""

sentiment_prompt = ChatPromptTemplate.from_messages([
    ("system", sentiment_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", """
User's message: {user_input}

Sentiment Analysis Results:
- Overall Sentiment: {sentiment}
- Polarity Score: {polarity} (range: -1 to 1)
- Subjectivity Score: {subjectivity} (range: 0 to 1)
- Detected Emotion: {emotion}
- Confidence Level: {confidence}

Please provide a thoughtful analysis and response to the user based on this information.
""")
])

# Session History Function
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# Create the conversational chain
chain = sentiment_prompt | llm
conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="user_input",
    history_messages_key="chat_history"
)

# Main chat interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat Interface")
    
    # Display chat history
    if session_id in st.session_state.store:
        session_history = st.session_state.store[session_id]
        for msg in session_history.messages:
            with st.chat_message(msg.type):
                st.markdown(msg.content)
    
    # User input
    if user_input := st.chat_input("Share your thoughts or feelings..."):
        # Analyze sentiment
        sentiment_data = analyze_sentiment_textblob(user_input)
        emotion = get_emotion_from_sentiment(sentiment_data["polarity"], sentiment_data["subjectivity"])
        
        # Store sentiment data
        sentiment_record = {
            "timestamp": datetime.now(),
            "text": user_input,
            "sentiment": sentiment_data["sentiment"],
            "polarity": sentiment_data["polarity"],
            "subjectivity": sentiment_data["subjectivity"],
            "emotion": emotion,
            "confidence": sentiment_data["confidence"]
        }
        st.session_state.sentiment_history.append(sentiment_record)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing sentiment..."):
                try:
                    response = conversational_chain.invoke(
                        {
                            "user_input": user_input,
                            "sentiment": sentiment_data["sentiment"],
                            "polarity": round(sentiment_data["polarity"], 3),
                            "subjectivity": round(sentiment_data["subjectivity"], 3),
                            "emotion": emotion,
                            "confidence": round(sentiment_data["confidence"], 3)
                        },
                        config={"configurable": {"session_id": session_id}}
                    )
                    st.markdown(response.content)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Please check your API configuration.")

# Sidebar - Current Analysis
with col2:
    if st.session_state.sentiment_history:
        latest_analysis = st.session_state.sentiment_history[-1]
        
        st.subheader("📈 Current Analysis")
        
        # Sentiment indicator
        sentiment_color = {
            "Positive": "🟢",
            "Negative": "🔴", 
            "Neutral": "🟡"
        }
        
        st.metric(
            label="Sentiment",
            value=f"{sentiment_color.get(latest_analysis['sentiment'], '⚪')} {latest_analysis['sentiment']}",
            delta=f"Confidence: {latest_analysis['confidence']:.2f}"
        )
        
        st.metric(
            label="Emotion",
            value=latest_analysis['emotion']
        )
        
        if show_detailed_analysis:
            st.subheader("🔍 Detailed Scores")
            
            # Polarity gauge
            fig_polarity = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = latest_analysis['polarity'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Polarity"},
                gauge = {
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.1], 'color': "lightcoral"},
                        {'range': [-0.1, 0.1], 'color': "lightgray"},
                        {'range': [0.1, 1], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ))
            fig_polarity.update_layout(height=200)
            st.plotly_chart(fig_polarity, use_container_width=True)
            
            # Subjectivity bar
            st.progress(latest_analysis['subjectivity'], text=f"Subjectivity: {latest_analysis['subjectivity']:.2f}")

# Sentiment trend analysis
if show_sentiment_trend and len(st.session_state.sentiment_history) > 1:
    st.subheader("📊 Sentiment Trend")
    
    df = pd.DataFrame(st.session_state.sentiment_history)
    
    # Time series plot
    fig_trend = px.line(
        df, 
        x='timestamp', 
        y='polarity',
        title='Sentiment Over Time',
        color_discrete_sequence=['#1f77b4']
    )
    fig_trend.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_trend.update_layout(height=300)
    st.plotly_chart(fig_trend, use_container_width=True)

# Emotion breakdown
if show_emotion_breakdown and st.session_state.sentiment_history:
    st.subheader("🎭 Emotion Distribution")
    
    df = pd.DataFrame(st.session_state.sentiment_history)
    emotion_counts = df['emotion'].value_counts()
    
    fig_emotions = px.pie(
        values=emotion_counts.values,
        names=emotion_counts.index,
        title="Emotions in This Session"
    )
    fig_emotions.update_layout(height=300)
    st.plotly_chart(fig_emotions, use_container_width=True)

# Footer
if st.sidebar.button("Clear Chat History"):
    if session_id in st.session_state.store:
        del st.session_state.store[session_id]
    st.session_state.sentiment_history = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ for emotional wellbeing By itsBabuaa")