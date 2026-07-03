import os
import smtplib
from email.mime.text             import MIMEText
from typing                      import TypedDict, Annotated
from dotenv                      import load_dotenv
from langchain_openai            import ChatOpenAI
from langchain_core.messages     import HumanMessage, AIMessage, SystemMessage
from langgraph.graph             import StateGraph, START, END
from vector_db                   import query_vector_db
from langchain_community.tools   import DuckDuckGoSearchRun 
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base        import BaseStore
from pydantic                    import BaseModel, Field
from langchain_openai            import ChatOpenAI

load_dotenv()

MODEL_NAME = 'llama3.2:3b'


def send_secure_email(recipient:str, subject:str, body:str):
    sender_email    = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    
    msg             = MIMEText(body)
    msg['Subject']  = subject
    msg['From']     = sender_email
    msg['To']       = recipient

    print(f"[SYSTEM LOCK]: Connecting to SMTP server secure channel...")
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient, msg.as_string())
    print(f"Email Successfully send to {recipient}!")



'''
STATE:
'''
class AgentState(TypedDict):
    user_input: str
    retrieved_context : str
    messages: list

class UserProfileFact(BaseModel):
    subject:str        = Field(description="The entity the fact is about (e.g., 'User')") 
    preference_key:str = Field(description="The category of preference (e.g., 'operating)_system', 'hardware', 'coding_style')")
    value:str          = Field(description="The core fact extracted (e.g., 'Linux mint', 'Ryzen 5 3500 PC')")

'''
NODES: TAKES THE CURRENT STATE AS AN INPUT, DOES SOME WORK, AND RETURNS AN
UPDATED PIECE OF THAT STATE:
'''
def retrieve_node(state:AgentState) -> dict:
    print(f"\n[Graph Node]: Querying ChromaDB index...")
    context = query_vector_db(state['user_input'], num_results=1)
    print(f"[Graph Node]: Context Retrieved.")
    return {"retrieved_context": context}

'''
ANOTHER NODE: 
THIS GENERATES A RESPONSE:
'''
def generate_node(state: AgentState) -> dict:
    print(f"[Graph Node]: Streaming to {MODEL_NAME}...")

    llm = ChatOpenAI(
        base_url    = "http://localhost:11434/v1",
        api_key     = "ollama",
        model       = MODEL_NAME,
        temperature = 1.0
    )

    if state["retrieved_context"] and state["retrieved_context"] != "No matching context found.":
        formatted_user_payload = (
            f"Context snippet from database:\n{state['retrieved_context']}\n\n"
            f"User Question : {state['user_input']}"
        )
    else:
        formatted_user_payload = state['user_input']

    # formatted_user_payload = (
    #     f"Context snipped from database:\n{state['retrieved_context']}\n\n"
    #     f"User Question: {state['user_input']}"
    # )

    system_prompt = SystemMessage(
        content=f"You are local AI agent running on {MODEL_NAME} with persistent memory."
    )

    chat_history = state.get("messages",[])
    prompt_messages = [system_prompt] + chat_history + [HumanMessage(content=formatted_user_payload)]

    print("\nAgent: ", end="", flush=True)
    agent_reply = ""
    for chunk in llm.stream(prompt_messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            agent_reply += chunk.content
    print("\n")
    
    # Update the historical log array with both turns for future loops
    updated_messages = chat_history + [
        HumanMessage(content=formatted_user_payload),
        AIMessage(content=agent_reply)
    ]
    
    return {"messages": updated_messages}

def web_search_node(state:AgentState) -> dict:
    print(f"\nGraph Node : Local DB dry, Launcing live internet search:...")
    search = DuckDuckGoSearchRun()
    search_result = search.run(state["user_input"])
    print(f"[Graph Node]: Web results compiled.")
    return {"retrieved_context":search_result}

def memory_extraction_node(state:AgentState):
    recent_chat = state['messages'][-2:]
    llm = ChatOpenAI(model=f"{MODEL_NAME}", base_url = 'http://localhost:11434/v1', api_key='ollama')
    structured_llm = llm.with_structured_output(UserProfileFact)
    prompt = (
        "You are a memory processor. Analyze the following recent dialogue. "
        "IF and ONLY IF the user explicitly states a permanent preference, core habit, "
        "their technical stack, hardware config, or personal anchor details, extract it. "
        "Ignore transient casual chit-chat, greetings, or temporary issues.\n\n"
       f"Recent conversation:\n{recent_chat}"
    )

    try:
        extracted:UserProfileFact = structured_llm.invoke(prompt)
        if extracted:
            extracted_facts = {
                extracted.preference_key:extracted.value
            }
            print(f"[Memory Agent]: Found permanent fact! -> {extracted_facts}")
    except Exception:
        pass

    return {}

def secure_email_sending_node(state:AgentState) -> dict:
    last_message = state['messages'][-1].content if state['messgaes'] else ''
    if "[TRIGGER_EMAIL]" in last_message:
        print("\nSECURITY ALERT: Agent Requesting Email Outbound Dispatch")
        print("Proposed Content:")
        print(f"---Outgoing messgae---\n{last_message}\n----------END------")

        user_auth = input("\nDo you authorize sending this email ? (Y/N): ")

        if user_auth.strip().lower() == 'y':
            try:
                send_secure_email(
                    recipient='test-target@domain.com',
                    subject="Automated Agent Update",
                    body=last_message
                )
            except Exception as e:
                print(f"Email system failed: {e}")
        else:
            print("Access Denied! Execution aborted by user security protocol.")
    return {}


'''
NORMAL SHIT : ASK QUESTION -> PULL CONTEXT MATCHING PART FROM VECTOR DB -> SEND BOTH TO LLM
              WHAT IF THERE'S NO MATCHING CONTEXT ? PYTHON DUMBASS WILL JUST ATTACH EMPTY CONTEXT AND SEND IT
THE FOLLOWING : IF NOT MATCHING CONTEXT SWITCH TO GENERAL KNOWLEDGE THAT THE MODEL WAS TRAINED ON:
'''
def check_context_router(state:AgentState) -> str:
    context = state.get("retrieved_context","").strip()
    if not context or context == 'No matching context found':
        print("[Graph Router]:No useful database context found. Switching to general knowledge.")
        return 'use_general_knowledge'

    print("[Graph Router]:Valid context found. Using standard generation.")
    return 'use_rag_context'

'''
THIS FUNCTION CREATES A WORKFLOW : 
S1 -> ADD A NODE THAT RETRIEVES THE CONTEXT.
S2 -> ADD A NODE THAT GENERATES A RESPONSE.
S3 -> ADD THE EDGES AKA PATHS : 
        START --> RETRIEVE CONTEXT --> GENERATE RESONSE --> END
S4 -> RETURN THE COMPILED WORKFLOW:
S5 -> INSIDE THE RUN_AGENT_LOOP WE WILL USE THIS WORKFLOW TO RUN THE AGENT (YO AGENT DAWG, THIS IS YOUR WORKFLOW STFU AND FOLLOW THESE STEPS)
'''
def compile_agent_graph():
    """Wires up the state machines nodes and execution directions."""
    workflow = StateGraph(AgentState)
    
    # Register our worker nodes
    workflow.add_node("retrieve_context_step", retrieve_node)
    workflow.add_node("generate_response_step", generate_node)

    workflow.add_node("web_search_step", web_search_node)

    workflow.add_node("extract_memory_step", memory_extraction_node)
    
    # START --> RETRIEVE CONTEXT RELATED TO THE QUESTION SENT BY USER:
    workflow.add_edge(START, "retrieve_context_step")                    # Step 1: Start goes straight to Retrieval

    # AFTER RETRIEVING --> GENERATE RESPONSE IF CONTEXT AVAILABLE OR USE GENERAL KNOWLEDGE TO CREATE RESPONSE:
    workflow.add_conditional_edges(                         
        "retrieve_context_step",
        check_context_router,
        {
            "use_rag_context"     : "generate_response_step",
            "use_general_knowlege": "web_search_step"
        }
    )

  # workflow.add_edge("retrieve_context_step", "generate_response_step") # Step 2: Retrieval transfers over to Generation

    # IF NOT CONTEXT MATCHING --> DO WE SEARCH --> THEN GENERATE RESPONSE :   
    workflow.add_edge("web_search_step", "generate_response_step")

    workflow.add_edge('generate_response_step', "extract_memory_step")

    # --> END
    workflow.add_edge("generate_response_step", END)                     # Step 3: Generation stops execution
    
    return workflow.compile()

# 4. USER CONSOLE RUNTIME LOOP
def run_agent_loop():
    print(f"--- LangGraph Orchestrated Local Agent Active [Model: {MODEL_NAME}] ---")
    print("Type 'exit' to shut down the script.\n")
    
    # Compile the state graph blueprint
    app = compile_agent_graph()
    
    # Track messages locally in memory during this console session
    conversation_history = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        if not user_input.strip():
            continue
            
        # Invoke our graph engine. 
        # We pass the current input and the session's ongoing conversation log
        graph_output = app.invoke({
            "user_input": user_input,
            "messages": conversation_history
        })
        
        # Keep the updated history state returned by the graph loop for the next user turn
        conversation_history = graph_output["messages"]

if __name__ == "__main__":
    run_agent_loop()