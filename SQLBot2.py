# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:24:00 2024

@author: karan
"""

import streamlit as st
import os
import numpy as np
import pandas as pd
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# Initialize
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API"]
    
# Comment out the below to opt-out of using LangSmith in this notebook. Not required.
if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

#Initialise SQLlite
df= pd.read_csv(r'\s1.csv')

# Inspect the first few rows of the DataFrame
#print(df.head())

# Rename columns
df.rename(columns={
    'Invoice ID': 'invoice_id',
    'Branch': 'branch',
    'City': 'city',
    'Customer type': 'customer_type',
    'Gender': 'gender',
    'Product line': 'product_line',
    'Unit price': 'unit_price',
    'Quantity': 'quantity',
    'Tax 5%': 'tax_5',
    'Total': 'total',
    'Date': 'date',
    'Time': 'time',
    'Payment': 'payment',
    'cogs': 'cogs',
    'gross margin percentage': 'gross_margin_percentage',
    'gross income': 'gross_income',
    'Rating': 'rating'
}, inplace=True)

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('example.db')
c = conn.cursor()

# Create table
c.execute('''
CREATE TABLE IF NOT EXISTS sales_data (
    invoice_id TEXT,
    branch TEXT,
    city TEXT,
    customer_type TEXT,
    gender TEXT,
    product_line TEXT,
    unit_price REAL,
    quantity INTEGER,
    tax_5 REAL,
    total REAL,
    date TEXT,
    time TEXT,
    payment TEXT,
    cogs REAL,
    gross_margin_percentage REAL,
    gross_income REAL,
    rating REAL
)
''')

# Insert data into the table
for row in df.itertuples(index=False):
    c.execute('''
    INSERT INTO sales_data (invoice_id, branch, city, customer_type, gender, product_line, unit_price, quantity, tax_5, total, date, time, payment, cogs, gross_margin_percentage, gross_income, rating)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (row.invoice_id, row.branch, row.city, row.customer_type, row.gender, row.product_line, row.unit_price, row.quantity, row.tax_5, row.total, row.date, row.time, row.payment, row.cogs, row.gross_margin_percentage, row.gross_income, row.rating))

# Commit the changes and close the connection
conn.commit()
conn.close()

db = SQLDatabase.from_uri("sqlite:///example.db")

#Function to read sql query
def read_sql_query(sql, db):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    for row in rows:
        print(row)
    conn.close()

#GenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question in a formal way.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

answer = answer_prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)


# Streamlit app layout
st.title("Sales Chatbot")
st.write("Ask me anything about the sales data!")

# Input text for user query
user_query = st.text_input("Your Question:")

def generate_response(query_user):
    return chain.invoke({"question": query_user})  
    

if user_query:
    with st.spinner("Generating response..."):
        response = generate_response(user_query)
        st.write("### Response:")
        st.write(response)
