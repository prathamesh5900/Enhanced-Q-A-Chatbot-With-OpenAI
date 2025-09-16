import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# LangSmith tracking (optional)
#os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_PROJECT"] = "Q&A chatbot with OpenAI"

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the queries."),
        ("user", "Question: {question}")
    ]
)

# Function to generate response
def generate_response(question, api_key, model, temperature, max_tokens):
    llm_instance = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key  # Passed API key directly here
    )
    output_parser = StrOutputParser()
    chain = prompt | llm_instance | output_parser
    answer = chain.invoke({"question": question})
    return answer

# App title
st.title("Enhanced Q&A Chatbot With OpenAI")

# Sidebar settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
selected_model = st.sidebar.selectbox("Select an OpenAI model", ["gpt-4o", "gpt-4o-turbo", "gpt-4"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    if api_key.strip() == "":
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        response = generate_response(user_input, api_key, selected_model, temperature, max_tokens)
        st.write(response)
else:
    st.write("Please provide the query")

