import requests
import streamlit as st
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from datetime import datetime, timedelta
from langchain.chains import RetrievalQA
import json
import os
import requests
from typing import List, Dict
import time


GOOGLE_API_KEY = "AIzaSyDUhEbbVwCyQwm79C3_ncsAvpysHd7J6sg"
BASE_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"

def fetch_places(destination, interest):
    """Fetch top places related to the interest in the given location using Google Places API"""
    query = f"{interest} in {destination}"
    params = {"query": query, "key": GOOGLE_API_KEY}
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json().get("results", [])
        places = [{"name": place["name"], "address": place.get("formatted_address", "Unknown")} for place in results[:5]]
        return places
    return []


def initialize_llm():
    return Ollama(model="llama3.2", temperature=0.7)

def initialize_embeddings():
    return OllamaEmbeddings(model="llama3.2")

llm = initialize_llm()
embeddings = initialize_embeddings()

def initialize_vectorstore(destination):
    """Creates or loads FAISS vector store for a given destination"""
    vectorstore_path = f"vectorstore_{destination.replace(' ', '_')}"
    
    if os.path.exists(vectorstore_path):
        return FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    
    places = fetch_places(destination, "Tourist attractions")
    text_data = "\n".join([f"{p['name']} - {p['address']}" for p in places])
    
    vectorstore = FAISS.from_texts([text_data], embeddings)
    vectorstore.save_local(vectorstore_path)
    return vectorstore


def generate_detailed_itinerary(destination: str, interests: List[str], num_days: int) -> List[Dict]:
    """Generate a detailed itinerary using dynamic data"""
    
    vectorstore = initialize_vectorstore(destination)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    prompt_template = PromptTemplate(
        input_variables=["destination", "interests", "day_num"],
        template="""
        You are a travel expert. Generate a detailed Day {day_num} itinerary for {destination} focused on {interests}.
        Include:
        - Morning, Afternoon, and Evening activities
        - Local landmarks, food places, and cultural spots
        - Best times for visits
        
        Format as JSON:
        {{
            "morning": {{"time": "time", "activity": "activity", "description": "description"}},
            "afternoon": {{"time": "time", "activity": "activity", "description": "description"}},
            "evening": {{"time": "time", "activity": "activity", "description": "description"}}
        }}
        """
    )

    itinerary = []
    
    for day in range(1, num_days + 1):
        prompt = prompt_template.format(destination=destination, interests=", ".join(interests), day_num=day)
        response = qa_chain.run(prompt)
        
        try:
            itinerary.append(json.loads(response.strip()))
        except json.JSONDecodeError:
            itinerary.append({
                "morning": {"time": "09:00", "activity": "Explore city", "description": "Walk around the city center."},
                "afternoon": {"time": "14:00", "activity": "Visit local attractions", "description": "Discover local history and culture."},
                "evening": {"time": "19:00", "activity": "Try local cuisine", "description": "Dine at a famous local restaurant."}
            })

    return itinerary

def format_itinerary_markdown(destination: str, itinerary: List[Dict], start_date: datetime) -> str:
    """Convert itinerary into markdown format"""
    markdown = f"# 🌍 Travel Itinerary for {destination}\n\n"
    current_date = start_date

    for day_num, day in enumerate(itinerary, 1):
        markdown += f"## Day {day_num} - {current_date.strftime('%A, %B %d, %Y')}\n\n"
        for period in ["morning", "afternoon", "evening"]:
            if period in day:
                markdown += f"### 🕒 {day[period]['time']} - {day[period]['activity']}\n"
                markdown += f"{day[period]['description']}\n\n"
        markdown += "---\n\n"
        current_date += timedelta(days=1)

    return markdown


def set_premium_styles():
    """Apply premium styling to the application"""
    st.markdown(
        """
        <style>
        /* Premium glass morphism style */
        .glassmorphism {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 20px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            transition: all 0.3s ease;
        }
        
        .glassmorphism:hover {
            box-shadow: 0 8px 36px 0 rgba(31, 38, 135, 0.25);
            transform: translateY(-3px);
        }
        
        /* Button styles */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 500;
            border: none;
            padding: 10px 25px;
            border-radius: 25px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
        }
        
        /* Form input fields */
        .stTextInput > div > div > input,
        .stDateInput > div > div > input,
        .stSelectbox > div > div > div {
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            font-weight: 400;
            padding: 8px 12px;
            transition: all 0.2s ease;
        }
        
        /* Background overlay */
        .stApp {
            background-image: linear-gradient(to bottom, rgba(0, 0, 30, 0.7), rgba(0, 0, 30, 0.5)), url("https://www.traveltrendstoday.in/wp-content/uploads/2024/11/IATA-Airplane.jpg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        
        /* Heading animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        h1, h2, h3 {
            color: white;
            animation: fadeIn 0.8s ease-out forwards;
        }
        
        /* Text styling */
        p, label, .stMarkdown {
            color: rgba(255, 255, 255, 0.9);
        }
        
        /* Content card styling */
        .content-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        
        .content-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        }
        
        /* Loader animation */
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        
        .stSpinner > div {
            animation: pulse 1.5s infinite ease-in-out;
        }
        
        /* Logo animation */
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        .logo {
            animation: float 5s ease-in-out infinite;
            display: inline-block;
        }
        
        /* Success message styling */
        .success-message {
            background: rgba(46, 204, 113, 0.2);
            border-left: 4px solid #2ecc71;
            border-radius: 4px;
            padding: 10px 15px;
            margin: 10px 0;
        }
        
        /* Error message styling */
        .error-message {
            background: rgba(231, 76, 60, 0.2);
            border-left: 4px solid #e74c3c;
            border-radius: 4px;
            padding: 10px 15px;
            margin: 10px 0;
        }
        
        /* Download button styling */
        .download-btn {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            padding: 8px 15px;
            color: white;
            transition: all 0.3s ease;
        }
        
        .download-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def animate_loading():
    """Show an animated loading sequence"""
    progress_text = "Preparing your premium experience..."
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)
    progress_bar.empty()

# Initialize the app with premium styling
st.set_page_config(page_title="ExpediAI Premium", layout="wide")
set_premium_styles()

# Animated intro (only on first load)
if 'initialized' not in st.session_state:
    animate_loading()
    st.session_state.initialized = True

# Title with animation
st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1><span class="logo">🌍</span> Expedi<span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AI</span> Premium <span class="logo">✈️</span></h1>
        <p style="font-size: 1.2em; opacity: 0.8;">Your Personalized Luxury Travel Experience</p>
    </div>
""", unsafe_allow_html=True)

# Main container with glassmorphism effect
st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)

# Two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌆 Destination")
    destination = st.text_input("", "Paris", placeholder="Enter your dream destination")
    
    st.markdown("### 📅 Travel Dates")
    start_date = st.date_input("Start Date", datetime.now())
    num_days = st.slider("Duration (Days)", 1, 30, 3)

with col2:
    st.markdown("### 🎯 Travel Preferences")
    interests = st.multiselect(
        "Select your travel interests",
        ["Historical Sites", "Museums", "Local Cuisine", "Shopping", "Nature", "Adventure", "Art & Culture", "Nightlife", "Luxury", "Relaxation"],
        ["Historical Sites", "Local Cuisine"]
    )

st.markdown('</div>', unsafe_allow_html=True)

# Button with custom styling
st.markdown('<div style="text-align: center; margin: 30px 0;">', unsafe_allow_html=True)
generate_button = st.button("✨ Create My Premium Itinerary")
st.markdown('</div>', unsafe_allow_html=True)

# Initialize itinerary as None
itinerary = None

if generate_button:
    if not interests:
        st.markdown('<div class="error-message">Please select at least one interest to personalize your journey.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Crafting your bespoke travel experience..."):
            try:
                # Add artificial delay for premium feeling
                time.sleep(1)
                
                itinerary = generate_detailed_itinerary(destination, interests, num_days)
                markdown_content = format_itinerary_markdown(destination, itinerary, datetime.combine(start_date, datetime.min.time()))
                
                # Display the itinerary with styling
                st.markdown('<div class="content-card">', unsafe_allow_html=True)
                st.markdown(markdown_content)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Download options with improved styling
                st.markdown('<div style="display: flex; justify-content: center; gap: 20px; margin: 30px 0;">', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📝 Download Markdown Itinerary",
                        markdown_content,
                        file_name=f"{destination}_luxury_itinerary.md",
                        mime="text/markdown",
                        help="Get your itinerary in a beautifully formatted markdown document"
                    )
                with col2:
                    st.download_button(
                        "🔄 Download as JSON",
                        json.dumps(itinerary, indent=2),
                        file_name=f"{destination}_luxury_itinerary.json",
                        mime="application/json",
                        help="Get your itinerary data in JSON format"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            
            except Exception as e:
                st.markdown(f'<div class="error-message">We apologize, but an error occurred: {str(e)}</div>', unsafe_allow_html=True)

# Email subscription section
FORMSPREE_ENDPOINT = "https://formspree.io/f/xpwqodlb"

def send_email(email: str, itinerary_text: str) -> bool:
    """Send the itinerary to the user via Formspree."""
    data = {
        "email": email,
        "message": itinerary_text
    }
    
    response = requests.post(FORMSPREE_ENDPOINT, data=data)
    
    if response.status_code == 200:
        return True
    return False

# Email form with premium styling
st.markdown('<div class="glassmorphism" style="margin-top: 40px;">', unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h3>📩 Receive Your Premium Itinerary</h3>
        <p>Get your personalized travel plan delivered straight to your inbox</p>
    </div>
""", unsafe_allow_html=True)

with st.form("email_form"):
    email = st.text_input("", placeholder="Enter your email address")
    submit_email = st.form_submit_button("📤 Send to My Inbox")

    if submit_email:
        if not email:
            st.markdown('<div class="error-message">Please provide a valid email address.</div>', unsafe_allow_html=True)
        elif itinerary is None:
            st.markdown('<div class="error-message">Please generate an itinerary first before requesting email delivery.</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Preparing your personalized email..."):
                # Add artificial delay for premium feeling
                time.sleep(1.5)
                itinerary_text = format_itinerary_markdown(destination, itinerary, datetime.combine(start_date, datetime.min.time()))
                if send_email(email, itinerary_text):
                    st.markdown('<div class="success-message">✅ Your premium itinerary has been sent successfully! Please check your inbox.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-message">❌ We were unable to send your email at this time. Please try again later.</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer with subtle animation
st.markdown("""
    <footer style="text-align: center; margin-top: 60px; padding: 20px; opacity: 0.8;">
        <p>ExpediAI Premium — Crafting Unforgettable Journeys</p>
        <div style="margin: 15px 0;">
            <span style="margin: 0 10px;">✦</span>
            <span style="margin: 0 10px;">Built with Streamlit, LangChain & AI</span>
            <span style="margin: 0 10px;">✦</span>
        </div>
        <p style="font-size: 0.8em;">©2025 ExpediAI • All Rights Reserved</p>
    </footer>
""", unsafe_allow_html=True)