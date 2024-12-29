# import streamlit as st
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.llms import Ollama 

# import os
# from dotenv import load_dotenv
# load_dotenv()

# # Langsmith tracking
# os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACKING_V2"] = 'true'
# os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot with Ollama"

# # Prompt template 
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant. Please respond to the user queries"),
#         ("user", "Question: {question}")
#     ]
# )

# def generate_response(question, engine, temperature, max_tokens):
#     llm = Ollama(model=engine, temperature=temperature)
#     output_parser = StrOutputParser()  # Create an instance of StrOutputParser
#     chain = prompt | llm | output_parser
#     answer = chain.invoke({"question": question})
#     return answer


# # Streamlit app
# engine = st.sidebar.selectbox("Select Open AI model", ["llama3.2:latest", 'gemma2:2b'])

# # Adjust response parameters
# temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
# max_tokens = st.sidebar.slider("Max Tokens", min_value=0, max_value=300, value=150)

# # Main interface for user input
# st.write("Go ahead and ask any question")
# user_input = st.text_input("You: ")

# if user_input:
#     response = generate_response(user_input, engine, temperature, max_tokens)
#     st.write(response)
# else:
#     st.write("Please enter your question to use the chatbot")





import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

# 🌍 Load environment variables
# 🌍 Load environment variables
load_dotenv()

# api_key = os.getenv("LANGCHAIN_API_KEY")
api_key = st.secrets["LANGCHAIN_API_KEY"]

if api_key is None:
    st.error("⚠️ LANGCHAIN_API_KEY not found in environment variables!")
    raise ValueError("LANGCHAIN_API_KEY environment variable is not set")

os.environ['LANGCHAIN_API_KEY'] = api_key
os.environ["LANGCHAIN_TRACKING_V2"] = 'true'
os.environ["LANGCHAIN_PROJECT"] = "Ultimate AI Chatbot Pro"


# 🎨 Page Configuration
st.set_page_config(
    page_title="AI Chatbot Pro+",
    page_icon="🤖",
    layout="wide"
)

# 📝 Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_response" not in st.session_state:
    st.session_state.latest_response = ""

# 🧠 Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional AI assistant. Provide precise, insightful, and well-structured responses."),
    ("user", "Question: {question}")
])

# 📊 Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Configuration Panel")
    st.subheader("Model Settings")
    engine = st.selectbox("🧠 Select AI Model", ["llama3.2:latest", 'gemma2:2b'])
    temperature = st.slider("🌡️ Temperature", min_value=0.0, max_value=1.0, value=0.7)
    max_tokens = st.slider("🔢 Max Tokens", min_value=50, max_value=500, value=200)
    
    st.divider()
    st.subheader("Session Controls")
    clear_chat = st.button("🗑️ Clear Chat History")
    download_history = st.download_button(
        label="📥 Download Chat History",
        data="\n".join(
            [f"You: {c['user']}\nAI: {c['ai']}" for c in st.session_state.get("chat_history", [])]
        ),
        file_name="chat_history.txt",
        mime="text/plain"
    )

    st.divider()
    st.subheader("📊 Session Analytics")
    st.write(f"Total Questions Asked: **{len(st.session_state.chat_history)}**")
    if st.session_state.chat_history:
        avg_length = sum(len(c['ai']) for c in st.session_state.chat_history) / len(st.session_state.chat_history)
        st.write(f"Avg. Response Length: **{avg_length:.2f} characters**")

# 🤖 Response Generation Function
def generate_response(question, engine, temperature, max_tokens):
    try:
        llm = Ollama(model=engine, temperature=temperature)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        return chain.invoke({"question": question})
    except Exception as e:
        return f"❌ Error: {str(e)}"

# 🌟 Main Interface
st.title("🤖 Ultimate AI Chatbot Pro+")
st.write("Ask me anything, and I’ll deliver accurate, insightful answers!")

# 📝 User Input Section
with st.container():
    user_input = st.text_input(
        "💬 Your Question:", 
        key="user_input", 
        placeholder="Type your question here...", 
        help="Ask me anything, and I'll provide a helpful response."
    )
    send = st.button("🚀 Ask AI")

# 📢 Display Latest Response First
if send and user_input:
    with st.spinner("🤔 Thinking..."):
        response = generate_response(user_input, engine, temperature, max_tokens)
        st.session_state.latest_response = response
        st.session_state.chat_history.insert(0, {"user": user_input, "ai": response})
        st.rerun()

# 🌟 Show Latest Response
if st.session_state.latest_response:
    st.write("---")
    # st.write("## 🗨️ **Latest AI Response**")
    st.markdown(f"{st.session_state.latest_response}")

# 📚 Display Chat History Below
st.write("---")
st.write("## 📚 Chat History (Most Recent First)")

for chat in st.session_state.chat_history[:10]:  # Show recent 10 chats
    with st.expander(f"🧑 **You:** {chat['user']}", expanded=False):
        st.markdown(f"**🤖 AI:** {chat['ai']}")

# 🗑️ Clear Chat History
if clear_chat:
    st.session_state.chat_history = []
    st.session_state.latest_response = ""
    st.success("✅ Chat history cleared!")
    st.rerun()

# 📢 Footer
st.write("---")
st.caption("🧠 Powered by LangChain & Ollama | 💼 Built with Streamlit | 🚀 Designed for Professional Use")
