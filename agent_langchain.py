import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# 1. Setup: Secure API Key
# (Ensure your .streamlit/secrets.toml has GEMINI_API_KEY)
if "GEMINI_API_KEY" not in st.secrets:
    st.error("⚠️ API Key not found. Please add it to your secrets.toml file.")
    st.stop()

api_key = st.secrets["GEMINI_API_KEY"]

# 2. The Brain (LangChain Wrapper)
# We enable 'google_search_retrieval' here to give it the Search Tool natively
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    google_search_retrieval=True, # <--- The Built-in Search Hand
    temperature=0
)

# 3. The Prompt (Personality + History)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a talented secretary of latin descent, but you speak English. Your nickname for me is papasito or papi"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# 4. The Chain (Brain + Prompt)
chain = prompt | llm

# 5. The Memory (Streamlit Specific)
# This auto-connects to st.session_state
history = StreamlitChatMessageHistory(key="langchain_messages")

# 6. The "Traffic Cop" (Runnable with History)
# This wraps the chain to auto-inject memory
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,  # Always returns the same history for this session
    input_messages_key="question",
    history_messages_key="history",
)

# --- UI LOGIC ---

st.title("Ninasita Bebesita")

# Show existing chat messages
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

# Handle user input
if prompt_text := st.chat_input("What's on your mind?"):
    # 1. Display User Message
    st.chat_message("human").write(prompt_text)

    # 2. Generate Reply
    # Note: We don't need to manually append to history; LangChain does it.
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.invoke({"question": prompt_text}, config=config)
            st.write(response.content)