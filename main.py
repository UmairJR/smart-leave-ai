from dotenv import load_dotenv
from langchain_groq import ChatGroq
import streamlit as st
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import tool
from langchain.schema import HumanMessage
import requests
from PyPDF2 import PdfReader
import os

load_dotenv()

FAST_API_URL = os.environ["FAST_API_URL"]

# Initialize OpenAI LLM
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")


@tool
def extract_name(query: str) -> str:
    """Extracts the name from a user query and correct the typos and returns it format (e.g., 'John Doe')."""
    date_prompt = f"""
    Extract the name and correct if any typos from the following request:
    "{query}"
    Reply with only the name.
    """
    return llm.invoke([HumanMessage(content=date_prompt)]).content.strip()


@tool
def get_employee_by_name(employee_name: str):
    """Find employees matching a given name."""
    api_url = f"{FAST_API_URL}leave-request?employee_name={employee_name}"
    try:
        response = requests.get(api_url)
        data = response.json()
        if not data:
            return None
        # Case when multiple employees are found
        if isinstance(data, list):
            return {
                "employee_name": data["employee_name"],
                "employee_ids": data["employee_ids"]
            }
        # Case when a single employee is found
        return {
            "employee_name": data["employee_name"],
            "employee_ids": data["employee_ids"]
        }

    except Exception as e:
        return {"error": f"Error fetching employee: {str(e)}"}


# Tool: Extract Date from Query
@tool
def extract_leave_date(query: str) -> str:
    """Extracts leave dates from a user query and determines:
       - Single date: 'DD MMM' (e.g., '10 Feb')
       - Date range: 'DD MMM to DD MMM' (e.g., '10 Feb to 20 Feb')
       - Multiple separate dates: 'DD MMM, DD MMM' (e.g., '10 Feb, 5 Mar, 7 Apr')
    """
    date_prompt = f"""
    Analyze the request: "{query}"

    1. Determine whether the request mentions:
       - A **single leave date** (e.g., '10 Feb')
       - A **date range** (e.g., '10 Feb to 20 Feb')
       - **Multiple separate leave dates** (e.g., '10 Feb, 5 Mar, 7 Apr')

    2. Based on the above, extract the dates and format them correctly:
       - If a single date is mentioned, return it as 'DD MMM'.
       - If a date range is mentioned, return it as 'DD MMM to DD MMM'.
       - If multiple separate dates are mentioned, return them as a comma-separated list in 'DD MMM, DD MMM' format.

    Reply **only** with the extracted dates in the correct format.
    """
    system_message = f"🔄 Extracting leave date from: {query}"
    st.write(f":green-background[{system_message}]")
    return llm.invoke([HumanMessage(content=date_prompt)]).content.strip()


# Tool: Check Leave Balance from API
@tool
def check_leave(employee_id: int, requested_dates: str) -> str:
    """Checks an employee's leave balance using an API and returns the result."""
    api_url = f"{FAST_API_URL}leave/{employee_id}"
    system_message = f"🔄 Checking leave balance for Employee ID: {employee_id}"
    st.write(f":green-background[{system_message}]")
    try:
        response = requests.get(api_url)
        data = response.json()
        if "error" in data:
            return "❌ Employee not found."
        remaining = data["remaining_cl"]
        leave_days_prompt = f"""
            Analyze the following leave request: "{requested_dates}"

            1. If the request mentions a **single date**, return the number of days as 1.
            2. If the request mentions a **date range**, calculate the total number of days in the range.
            3. If the request mentions **multiple separate dates**, calculate the number of separate days requested.

            Reply only with the number of days.
            """
        leave_days = int(llm.invoke([HumanMessage(content=leave_days_prompt)]).content.strip())
        return f"✅ Leave Available! Remaining CL: {remaining}" if remaining >= leave_days else "❌ No casual leave balance left."
    except Exception as e:
        return f"❌ Error fetching data: {str(e)}"


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        # Initialize a PDF reader
        reader = PdfReader(file)
        # Extract text from the first page
        text = reader.pages[0].extract_text()
    return text


# Tool: Check Leave Policies
@tool
def fetch_policies(requested_dates: str) -> str:
    """Checks if company policies restrict leave on the requested date."""
    # pdff
    folder_path = os.path.join(os.getcwd(), 'asset')
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if pdf_files:
        pdf_path = os.path.join(folder_path, pdf_files[0])
        policies = extract_text_from_pdf(pdf_path)
    system_message = f"🔄 Checking leave policies"
    st.write(f":green-background[{system_message}]")

    policy_prompt = f"""
    Given these policies: {policies}
    Analyze the request: "{requested_dates}"
    1. If a **single date** is mentioned, check if leave is allowed for that date.
    2. If a **range of dates** is mentioned, check if leave is allowed for the full range.
    3. If **multiple separate dates** are mentioned, check if leave is allowed for each date. 
    Reply with:
    ✅ Positive OR ❌ Negative, followed by a short reason.
    """
    return llm.invoke([HumanMessage(content=policy_prompt)]).content.strip()


# Tool: Check Email Conflicts
@tool
def check_employee_email(employee_id: int, requested_dates: str) -> str:
    """Checks if the employee has a meeting or event on the requested date."""
    events = {
        1: {
            "value": [
                {
                    "subject": "Meeting with CEO",
                    "bodyPreview": "Plan for Q3",
                    "startDate": "2025-02-10",  # 10 Feb
                    "endDate": "2025-02-10"  # 10  Feb
                },
                {
                    "subject": "Budget Allocation",
                    "bodyPreview": "Funds distribution",
                    "startDate": "2025-02-11",  # 11 Feb
                    "endDate": "2025-02-11"  # # 11 Feb
                }
            ]
        },
        2: {
            "value": [
                {
                    "subject": "Team Progress Review",
                    "bodyPreview": "Weekly updates",
                    "startDate": "2025-03-02",  # 2 Mar
                    "endDate": "2025-03-02"  # 2 Mar
                },
                {
                    "subject": "Client Feedback",
                    "bodyPreview": "Improvements needed",
                    "startDate": "2025-03-04",  # 4 Mar
                    "endDate": "2025-03-04"  # 4 Mar
                }
            ]
        },
        3: {
            "value": []
        },
        4: {
            "value": [
                {
                    "subject": "Hiring Plans",
                    "bodyPreview": "New recruitments",
                    "startDate": "2025-04-07",  # 7 April
                    "endDate": "2025-04-07"  # 7 April
                }
            ]
        },
        5: {
            "value": [
                {
                    "subject": "Security Updates",
                    "bodyPreview": "New protocols",
                    "startDate": "2024-02-26",  # 26 Feb
                    "endDate": "2024-02-27"  # 27 Feb
                },
                {
                    "subject": "Tech Stack Upgrade",
                    "bodyPreview": "Software improvements",
                    "startDate": "2024-03-07",  # 7 Mar
                    "endDate": "2024-03-07"  # 7 Mar
                }
            ]
        }
    }
    event_lists = events.get(employee_id, [])
    system_message = f"🔄 Checking event conflicts for Employee ID {employee_id} on {requested_dates}"
    st.write(f":green-background[{system_message}]")
    event_prompt = f"""
    Given these scheduled events: {event_lists}
    Analyze the following request: "{requested_dates}"
    1. If a **single date** is mentioned, check if there is any event on that date.
    2. If a **date range** is mentioned, check if there are any events during the full range.
    3. If **multiple separate dates** are mentioned, check if there are any events on each of those dates.
    Reply with:
    ✅ Positive OR ❌ Negative, followed by a short reason.
    """
    return llm.invoke([HumanMessage(content=event_prompt)]).content.strip()


# Define Agent with Tools
tools = [extract_leave_date, check_leave, fetch_policies, check_employee_email]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent='structured-chat-zero-shot-react-description',
    verbose=True
)

# Streamlit UI - Display
st.set_page_config(page_title="SmartLeave AI", page_icon="📝", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: white;'>📝 SmartLeave AI</h1>
    <p style='text-align: center; font-size:18px; color: #7F8C8D;'>
    <b>Ask for leave, get instant answers.</b></p>
    <hr style="border: 1px solid #BDC3C7;">
    """,
    unsafe_allow_html=True
)

query = st.text_input(
    "**Enter your leave request:**",
    placeholder="Can John Doe take a leave on Feb 10th?",
    help="Type your leave request."
)

if "selected_employee" not in st.session_state:
    st.session_state["selected_employee"] = None

if st.button("🔎 Check Leave", key="name_request"):
    name = extract_name(query)
    result = get_employee_by_name(name)
    if result and "error" not in result:
        st.session_state["search_result"] = result
    else:
        st.session_state["search_result"] = None
        st.error("❌ Employee not found. Please check the name and try again.")

if "hide" not in st.session_state:
    st.session_state["hide"] = False

if "search_result" in st.session_state and st.session_state["search_result"]:
    result = st.session_state["search_result"]
    found_header = st.empty()
    card = st.empty()
    found_header.write(f"##### 🔹 Found {len(result['employee_ids'][0])} record(s). Select an Employee ID:")
    with card.container():
        cols = st.columns(len(result['employee_ids'][0]))
        for i, emp_id in enumerate(result['employee_ids'][0]):
            with cols[i]:
                st.write("**👤 Employee**")
                st.write(f"**Name:** {result['employee_name']}")
                st.write(f"**ID:** `{emp_id}`")
                if st.button(f"✅ Select {emp_id}", key=f"select_{emp_id}"):
                    st.session_state["selected_employee"] = emp_id
                    st.session_state["hide"] = True

if st.session_state["hide"]:
    found_header = st.empty()
    card = st.empty()
    st.session_state["hide"] = False

if st.session_state["selected_employee"]:
    st.chat_message("user").write(query)
    with st.spinner("Processing your request..."):
        response = agent.run(f"Employee ID: {st.session_state['selected_employee']}. {query}")
    st.chat_message("assistant").write(response)
    st.session_state["selected_employee"] = None

