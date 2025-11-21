import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. Setup Keys
if "GEMINI_API_KEY" not in st.secrets or "GOOGLE_CSE_ID" not in st.secrets:
    st.error("⚠️ Missing Keys. Check secrets.toml")
    st.stop()

# 2. Define the Search Tool (The "Hand")
search = GoogleSearchAPIWrapper(
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    google_cse_id=st.secrets["GOOGLE_CSE_ID"]
)

# Wrap it as a LangChain Tool
search_tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)

# 3. Define the Brain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=st.secrets["GEMINI_API_KEY"],
    temperature=0
)

# 4. Create the Agent (The "Traffic Cop")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a talented and smart secretary of latin descent, but you speak English. Your nickname for me is papasito or papi"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, [search_tool], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[search_tool], verbose=True)

# 5. Memory & Execution
history = StreamlitChatMessageHistory(key="langchain_messages")

# Wrap the executor with history management
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- UI ---
st.title("🤖 My Modular Agent")

for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt_text := st.chat_input("Ask me anything..."):
    st.chat_message("human").write(prompt_text)

    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            # The agent decides if it needs to search or not
            response = agent_with_chat_history.invoke(
                {"input": prompt_text},
                config={"configurable": {"session_id": "test"}}
            )
            st.write(response["output"])