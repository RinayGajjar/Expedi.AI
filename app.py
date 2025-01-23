import streamlit as st
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict
import pandas as pd

# Initialize Llama model and Ollama embeddings
def initialize_llm():
    return Ollama(
        model="llama3.2",
        temperature=0.7
    )

def initialize_embeddings():
    return OllamaEmbeddings(
        model="llama3.2"  # Using same model for embeddings
    )

# Initialize components
llm = initialize_llm()
embeddings = initialize_embeddings()

# Create or load the vector store
def initialize_vectorstore():
    """Initialize FAISS vector store with travel-related documents"""
    try:
        # Try to load existing vector store
        return FAISS.load_local("travel_vectorstore")
    except:
        # Create new vector store if none exists
        sample_text = """
        Paris is the capital of France, known for the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.
        Popular activities include visiting museums, trying French cuisine, and walking along the Seine.
        The best time to visit is spring (March to May) or fall (September to November).
        
        Tokyo is Japan's busy capital, mixing ultramodern and traditional elements.
        Visit the Senso-ji Temple, explore Akihabara electronics district, and try local sushi.
        Cherry blossom season (late March to early April) is especially beautiful.
        
        New York City comprises 5 boroughs: Manhattan, Brooklyn, Queens, The Bronx, and Staten Island.
        Major attractions include Central Park, Times Square, Empire State Building, and Broadway shows.
        The city is famous for its diverse food scene, museums, and shopping.
        """
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(sample_text)
        
        vectorstore = FAISS.from_texts(texts, embeddings)
        vectorstore.save_local("travel_vectorstore")
        return vectorstore

# Initialize vector store
vectorstore = initialize_vectorstore()

def generate_detailed_itinerary(
    destination: str,
    interests: List[str],
    num_days: int
) -> List[Dict]:
    """Generate detailed itinerary using Llama and RAG"""
    
    # Create the retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    
    # Creating a prompt template for generating daily activities
    daily_prompt_template = PromptTemplate(
        input_variables=["destination", "interests", "day_num"],
        template="""
        You are a knowledgeable travel planner. Create a detailed day {day_num} itinerary for {destination} 
        focusing on these interests: {interests}.
        
        Provide specific attractions, timings, and descriptions.
        
        Format your response EXACTLY as a JSON string with this structure:
        {{
            "morning": {{"time": "time", "activity": "activity", "description": "description"}},
            "afternoon": {{"time": "time", "activity": "activity", "description": "description"}},
            "evening": {{"time": "time", "activity": "activity", "description": "description"}}
        }}
        
        Be realistic with timings and activities. Include local context and practical details.
        """
    )
    
    itinerary = []
    
    for day in range(1, num_days + 1):
        # Generate daily activities
        daily_prompt = daily_prompt_template.format(
            destination=destination,
            interests=", ".join(interests),
            day_num=day
        )
        
        try:
            # Getting response from LangChain
            response = qa_chain.run(daily_prompt)
            
            # Clean the response to ensure valid JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:-3]  # Remove JSON code block markers
            
            # Parse the JSON response
            daily_activities = json.loads(response)
            itinerary.append(daily_activities)
            
        except (json.JSONDecodeError, Exception) as e:
            st.warning(f"Error parsing day {day} activities, using fallback schedule")
            # Fallback if JSON parsing fails
            itinerary.append({
                "morning": {
                    "time": "09:00",
                    "activity": f"Explore {destination} attractions",
                    "description": f"Visit popular sites in {destination} related to {interests[0] if interests else 'local culture'}"
                },
                "afternoon": {
                    "time": "14:00",
                    "activity": "Cultural activities",
                    "description": "Experience local culture through food and sightseeing"
                },
                "evening": {
                    "time": "19:00",
                    "activity": "Dinner and relaxation",
                    "description": "Try local cuisine at recommended restaurants"
                }
            })
    
    return itinerary

def format_itinerary_markdown(destination: str, itinerary: List[Dict], start_date: datetime) -> str:
    """Format the itinerary as a markdown document"""
    markdown = f"# 🌍 Travel Itinerary for {destination}\n\n"
    current_date = start_date
    
    for day_num, day in enumerate(itinerary, 1):
        markdown += f"## Day {day_num} - {current_date.strftime('%A, %B %d, %Y')}\n\n"
        
        for period in ["morning", "afternoon", "evening"]:
            if period in day:
                activity = day[period]
                markdown += f"### 🕒 {activity['time']} - {activity['activity']}\n"
                markdown += f"{activity['description']}\n\n"
        
        markdown += "---\n\n"
        current_date += timedelta(days=1)
    
    return markdown

# Streamlit UI
def main():
    # Add background image before other Streamlit elements
    set_background()


st.set_page_config(page_title="Local AI Travel Planner", layout="wide")
st.title("🌍 AI Travel Itinerary Generator ✈️")
st.caption("Powered by Llama 3.2 and Ollama")

def set_background():
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://www.traveltrendstoday.in/wp-content/uploads/2024/11/IATA-Airplane.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """, unsafe_allow_html=True)

# Add a status indicator for Ollama connection
try:
    # Test Ollama connection
    llm.invoke("Hi")
    st.success("✅ Connected to Ollama successfully")
except Exception as e:
    st.error("❌ Error connecting to Ollama. Please make sure Ollama is running with the llama2 model installed.")
    st.stop()

# User inputs
col1, col2 = st.columns(2)

with col1:
    destination = st.text_input("Where would you like to go?", "Paris")
    start_date = st.date_input("Start Date", datetime.now())
    num_days = st.slider("Number of Days", 1, 14, 3)

with col2:
    interests = st.multiselect(
        "What are your interests?",
        [
            "Historical Sites", "Museums", "Local Cuisine", 
            "Shopping", "Nature", "Adventure", "Art & Culture",
            "Nightlife", "Architecture", "Local Markets"
        ],
        ["Historical Sites", "Local Cuisine"]
    )

if st.button("Generate Itinerary"):
    if len(interests) < 1:
        st.error("Please select at least one interest")
    else:
        with st.spinner("Generating your personalized itinerary..."):
            try:
                # Generate itinerary
                itinerary = generate_detailed_itinerary(destination, interests, num_days)
                
                # Format as markdown
                markdown_content = format_itinerary_markdown(
                    destination, itinerary, datetime.combine(start_date, datetime.min.time())
                )
                
                # Display itinerary
                st.markdown(markdown_content)
                
                # Download options
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        "📝 Download Itinarary",
                        markdown_content,
                        file_name=f"{destination}_itinerary.md",
                        mime="text/markdown"
                    )
                
                with col2:
                    st.download_button(
                        "🔄 Download as JSON",
                        json.dumps(itinerary, indent=2),
                        file_name=f"{destination}_itinerary.json",
                        mime="application/json"
                    )
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Langchain, and Ollama")


# Add setup instructions in an expander
with st.expander("📋 Setup Instructions"):
    st.markdown("""
    ### Prerequisites:
    1. Install Ollama from [ollama.ai](https://ollama.ai)
    2. Pull the Llama 2 model:
    ```bash
    ollama pull llama2
    ```
    
    ### Installation:
    ```bash
    pip install streamlit langchain faiss-cpu pandas
    ```
    
    ### Running the app:
    1. Start Ollama in a terminal:
    ```bash
    ollama serve
    ```
    2. Run the Streamlit app in another terminal:
    ```bash
    streamlit run app.py
    ```
    """)
if __name__ == "__main__":
    main()