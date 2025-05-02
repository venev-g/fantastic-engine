import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import json

import os

# Set page configuration
st.set_page_config(page_title="Memory-Enhanced Gemini Chatbot", page_icon="🧠")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "entity_memory" not in st.session_state:
    st.session_state.entity_memory = {}

# Display header
st.title("🧠 Gemini Chatbot with Entity Memory")
st.markdown("I remember your name and preferences across conversations!")

# Setup Gemini model
@st.cache_resource
def get_gemini_model():
    api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-8b",
        google_api_key=api_key,
        temperature=0.7,
    )

# Automatic entity extraction and categorization function using Gemini with direct prompting
def extract_entities(text):
    api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-8b",
        google_api_key=api_key,
        temperature=0,  # Keep temperature low for consistent extraction
    )
    
    # Enhanced prompting approach with entity categorization
    extraction_prompt = f"""
    Extract and categorize the following information from the text:
    1. Personal Information:
       - Name: The user's name if mentioned
       - Age: The user's age if mentioned
       - Location: Where the user is from or lives
       - Occupation: The user's job or profession
    
    2. Preferences and Interests:
       - Likes: Things the user likes, loves, enjoys, or prefers
       - Dislikes: Things the user dislikes, hates, or wants to avoid
       - Hobbies: Activities the user enjoys doing
       - Media: Books, movies, TV shows, music the user mentions enjoying
       - Food: Food or drinks the user enjoys
    
    3. Personal Context:
       - Future Plans: Things the user plans to do
       - Problems: Issues or challenges the user is facing
       - Goals: Things the user wants to achieve

    Text: {text}

    Respond with a JSON object in this exact format:
    {{
        "personal_info": {{
            "name": "extracted name or null",
            "age": "extracted age or null",
            "location": "extracted location or null",
            "occupation": "extracted occupation or null"
        }},
        "preferences": {{
            "likes": ["like1", "like2", ...],
            "dislikes": ["dislike1", "dislike2", ...],
            "hobbies": ["hobby1", "hobby2", ...],
            "media": ["media1", "media2", ...],
            "food": ["food1", "food2", ...]
        }},
        "context": {{
            "future_plans": ["plan1", "plan2", ...],
            "problems": ["problem1", "problem2", ...],
            "goals": ["goal1", "goal2", ...]
        }}
    }}
    
    Include only the JSON, no other text. Use null for missing values and empty arrays [] for empty lists.
    Only include information that's explicitly mentioned in the text.
    """
    
    try:
        # Get extraction result as text
        result = llm.invoke(extraction_prompt).content
        
        # Parse the JSON string to get structured data
        # Clean the result to extract just the JSON part
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = result[json_start:json_end]
            extracted_data = json.loads(json_str)
        else:
            extracted_data = json.loads(result)
            
        # Format the result for our memory structure
        entities = {}
        
        # Process personal information
        if "personal_info" in extracted_data:
            for key, value in extracted_data["personal_info"].items():
                if value and value != "null":
                    entities[key] = value
        
        # Process preferences with sub-categories
        if "preferences" in extracted_data:
            for pref_category, items in extracted_data["preferences"].items():
                if items and len(items) > 0:
                    category_key = f"preferences_{pref_category}"
                    entities[category_key] = items
        
        # Process context information
        if "context" in extracted_data:
            for context_type, items in extracted_data["context"].items():
                if items and len(items) > 0:
                    context_key = f"context_{context_type}" 
                    entities[context_key] = items
        
        return entities
    except Exception as e:
        st.error(f"Error extracting entities: {str(e)}")
        return {}

# Update entity memory with categorized information
def update_memory(text):
    new_entities = extract_entities(text)
    for key, value in new_entities.items():
        if isinstance(value, list):
            # Handle list-type entities like preferences, contexts
            if key not in st.session_state.entity_memory:
                st.session_state.entity_memory[key] = []
                
            # Check for duplicates before adding
            for item in value:
                if item not in st.session_state.entity_memory[key]:
                    st.session_state.entity_memory[key].append(item)
        else:
            # Handle single value entities like name, age
            st.session_state.entity_memory[key] = value

# Create system prompt with enhanced memory context
def get_system_prompt():
    memory_context = []
    
    # Add personal information
    personal_fields = {
        "name": "The user's name is {}.",
        "age": "The user is {} years old.",
        "location": "The user is from/lives in {}.",
        "occupation": "The user works as {}."
    }
    
    for field, template in personal_fields.items():
        if field in st.session_state.entity_memory:
            memory_context.append(template.format(st.session_state.entity_memory[field]))
    
    # Add preferences with categories
    preference_categories = {
        "preferences_likes": "things they like",
        "preferences_dislikes": "things they dislike",
        "preferences_hobbies": "hobbies",
        "preferences_media": "media they enjoy",
        "preferences_food": "foods/drinks they enjoy"
    }
    
    for pref_key, pref_desc in preference_categories.items():
        if pref_key in st.session_state.entity_memory and st.session_state.entity_memory[pref_key]:
            memory_context.append(f"The user has mentioned {pref_desc}: {', '.join(st.session_state.entity_memory[pref_key])}.")
    
    # Add context information
    context_categories = {
        "context_future_plans": "plans",
        "context_problems": "problems/challenges",
        "context_goals": "goals"
    }
    
    for context_key, context_desc in context_categories.items():
        if context_key in st.session_state.entity_memory and st.session_state.entity_memory[context_key]:
            memory_context.append(f"The user has mentioned these {context_desc}: {', '.join(st.session_state.entity_memory[context_key])}.")
    
    memory_text = " ".join(memory_context)
    
    return f"""You are a friendly AI assistant that remembers details about the user.
{memory_text}

Refer to this information naturally in your responses when appropriate.
Don't list or repeat all the information you know, but use it to personalize your responses.
Answer the user's questions helpfully and conversationally."""

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Send a message"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Update entity memory based on user input
    update_memory(prompt)
    
    # Prepare messages for the model
    messages = [SystemMessage(content=get_system_prompt())]
    
    # Add chat history (last 5 messages for context)
    for msg in st.session_state.messages[-5:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    
    # Generate response
    model = get_gemini_model()
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.invoke(messages)
            st.markdown(response.content)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.content})

# Add debug section with categorized display
with st.expander("Debug: View Categorized Memory"):
    # Group and display by categories
    if 'name' in st.session_state.entity_memory or 'age' in st.session_state.entity_memory or 'location' in st.session_state.entity_memory or 'occupation' in st.session_state.entity_memory:
        st.subheader("📝 Personal Information")
        personal_info = {}
        for field in ['name', 'age', 'location', 'occupation']:
            if field in st.session_state.entity_memory:
                personal_info[field] = st.session_state.entity_memory[field]
        st.write(personal_info)
    
    # Display preferences
    preference_keys = [k for k in st.session_state.entity_memory.keys() if k.startswith('preferences_')]
    if preference_keys:
        st.subheader("💙 Preferences & Interests")
        for key in preference_keys:
            category = key.replace('preferences_', '').capitalize()
            st.write(f"**{category}:** {', '.join(st.session_state.entity_memory[key])}")
    
    # Display context
    context_keys = [k for k in st.session_state.entity_memory.keys() if k.startswith('context_')]
    if context_keys:
        st.subheader("🔍 Personal Context")
        for key in context_keys:
            category = key.replace('context_', '').replace('_', ' ').capitalize()
            st.write(f"**{category}:** {', '.join(st.session_state.entity_memory[key])}")

