# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:57:19 2024

@author: karan
"""

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv
import os


# Initialize Pinecone, Sentence-BERT, and OpenAI
pc = Pinecone(api_key=st.secrets["PC_API"])
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
openai.api_key = st.secrets["OPENAI_API']
index = pc.Index("salesdata")

# Streamlit app layout
st.title("Sales Chatbot")
st.write("Ask me anything about the sales data!")

# Input text for user query
user_query = st.text_input("Your Question:")

# Function to handle querying and generating response
def generate_response(query):
    # Embed the user's query
    query_embedding = model.encode([query])

    # Search Pinecone for the top 5 matches
    result = index.query(vector=query_embedding.tolist(), top_k=5,include_metadata=True)


    # Get context from the matched results
    context = "\n".join([match['metadata']['Description'] for match in result['matches']])

    # Generate a response using GPT-4
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Question: {query}\n\nContext: {context}\n\nAnswer:"}
        ],
        max_tokens=150
    )
    # Return the response
    return response.choices[0].message.content

# Display the response if a query is submitted
if user_query:
    with st.spinner("Generating response..."):
        response = generate_response(user_query)
        st.write("### Response:")
        st.write(response)
