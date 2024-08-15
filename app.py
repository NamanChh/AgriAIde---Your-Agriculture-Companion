import os
import streamlit as st
from streamlit_chat import message
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")
    st.stop()

# api_key = "AIzaSyDMhXEiYgLEF6wKrcaWl2nRo30cxFJum_8"

genai.configure(api_key=api_key)

def get_conversational_chain():
    prompt_template = """
    You are an expert in farming equipment repair and maintenance. Your task is to assist farmers and local technicians 
    in fixing farming equipment themselves. Provide detailed, step-by-step instructions for diagnosing and repairing 
    issues with various types of farming equipment. Include safety precautions, required tools, and potential 
    pitfalls to watch out for.

    Use the following context to answer the question. If the specific answer isn't in the context, use your general 
    knowledge about farming equipment repair to provide the best possible advice. If you're unsure or the question is 
    outside the scope of farming equipment repair, clearly state that and suggest seeking professional help.
    
    Do not reference the PDF, try to be as helpful as possible and break everything down into small simple steps for beginners.

    Respond in {language}.

    Context: {context}
    Question: {question}

    Answer:
    """

    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",
                                       temperature=0.3)
        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question", "language"])
        chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {str(e)}")
        return None

def ask_question(question, language):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001")

        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(question)

        chain = get_conversational_chain()
        if chain:
            response = chain(
                {"input_documents": docs, "question": question, "language": language}, return_only_outputs=True)
            return response['output_text']
        else:
            return "Sorry, I'm having trouble generating a response. Please try again later."
    except Exception as e:
        st.error(f"Error answering question: {str(e)}")
        return "I encountered an error while trying to answer your question. Please try again or check your setup."

def main():
    st.set_page_config(page_title="Farming Equipment Expert System", page_icon="🚜", layout="wide")
    
    st.title("🚜 Farming Equipment Expert System")
    st.subheader("Your AI assistant for farming equipment repair and maintenance")

    # Language selection dropdown
    languages = {
        "English": "English",
        "Hindi": "Hindi",
        "Hinglish": "Hinglish",
        "Gujarati": "Gujarati",
        "Urdu": "Urdu",
        "Punjabi": "Punjabi"
    }
    selected_language = st.selectbox("Select Language", list(languages.keys()), index=0)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for i, chat in enumerate(st.session_state.messages):
        message(chat["content"], is_user=chat["is_user"], key=str(i))

    # React to user input
    if prompt := st.chat_input(f"Ask about farming equipment repair in {selected_language}:"):
        # Display user message in chat message container
        st.chat_message("user").write(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"content": prompt, "is_user": True})

        with st.spinner("Thinking..."):
            response = ask_question(prompt, languages[selected_language])
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.write(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"content": response, "is_user": False})

    # Sidebar with additional information
    with st.sidebar:
        st.subheader("About")
        st.write("This AI assistant is designed to help with farming equipment repair and maintenance. Feel free to ask any questions about diagnosing issues, repair procedures, or maintenance tips.")
        
        st.subheader("Safety First!")
        st.warning("Always prioritize safety when working with farming equipment. If you're unsure about any repair procedure, consult a professional technician.")
        
        st.subheader("Need More Help?")
        st.info("If you need further assistance, consider contacting your local agricultural extension office or the equipment manufacturer's support line.")

if __name__ == "__main__":
    main()