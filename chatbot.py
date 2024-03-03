import os
import streamlit as st


from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

tweet_prompt = PromptTemplate.from_template(" {topic}.")

tweet_chain = LLMChain(llm=llm, prompt=tweet_prompt, verbose=True)

if __name__=="__main__":
    st.title("Chatbot By Bala")
    
    # Input field for user to enter topic
    topic = st.text_input("Enter topic:", "")
    
    # Button to generate tweet
    if st.button("Generate Tweet"):
        # Generate response using LLMChain
        resp = tweet_chain.run(topic=topic)
        # Display response
        st.write("Generated Tweet:")
        st.write(resp)