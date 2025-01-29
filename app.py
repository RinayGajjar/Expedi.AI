import streamlit as st
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from datetime import datetime, timedelta
import json
import os
import requests
from typing import List, Dict

# ============ GOOGLE PLACES API SETUP ============
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

# ============ INITIALIZE OLLAMA & FAISS ============
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

# ============ GENERATE ITINERARY ============
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

# ============ STREAMLIT UI ============
def set_background(image_url: str):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
st.set_page_config(page_title="ExpediAI", layout="wide")
set_background("https://www.traveltrendstoday.in/wp-content/uploads/2024/11/IATA-Airplane.jpg")
st.title("🌍 Expedi AI ✈️")
st.caption("Your AI-Powered Travel Planner")

# UI Input Fields
col1, col2 = st.columns(2)

with col1:
    destination = st.text_input("Where would you like to go?", "Paris")
    start_date = st.date_input("Start Date", datetime.now())
    num_days = st.slider("Number of Days", 1, 14, 3)

with col2:
    interests = st.multiselect(
        "What are your interests?",
        ["Historical Sites", "Museums", "Local Cuisine", "Shopping", "Nature", "Adventure", "Art & Culture", "Nightlife"],
        ["Historical Sites", "Local Cuisine"]
    )

if st.button("Generate Itinerary"):
    if not interests:
        st.error("Please select at least one interest.")
    else:
        with st.spinner("Generating itinerary..."):
            try:
                itinerary = generate_detailed_itinerary(destination, interests, num_days)
                markdown_content = format_itinerary_markdown(destination, itinerary, datetime.combine(start_date, datetime.min.time()))
                
                st.markdown(markdown_content)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("📝 Download Itinerary", markdown_content, file_name=f"{destination}_itinerary.md", mime="text/markdown")
                with col2:
                    st.download_button("🔄 Download as JSON", json.dumps(itinerary, indent=2), file_name=f"{destination}_itinerary.json", mime="application/json")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

st.markdown("---")
st.markdown("Built with Streamlit, LangChain, FAISS, and Google Places API")

