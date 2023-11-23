import streamlit as st
from openai import OpenAI
import pymoo as pymoo
import os

st.title("Optimization Assistant")
st.write(
    "This is an optimization assistant that helps domain experts define, select, and interpret multi-objective optimization problems."
)

st.header("Step 1: Define the problem")

# Add drop down menu for problem area
problem_area = st.selectbox(
    "Select a problem area:",
    ["Energy", "Water", "Transportation", "Manufacturing", "Supply Chain"])

# Add drop down menu for problem type
problem_type = st.selectbox("Select a problem type:",
                            ["Strategic Planning", "Operations"])

# Add checkboxes for optimization objectives
objectives = st.multiselect(
    "Select optimization objectives:",
    ["Minimize Cost", "Maximize Benefits", "Minimize Risk"])

# Setup OpenAI assistant to guide user through optimization problem
client = OpenAI()
assistant = client.beta.assistants.create(
    name="Optimization Assistant",
    instructions=
    "You are an expert at leveraging multi-objective optimization to improve planning and operations of complex networks. You will help domain experts define problems, select methods, run, and interpret results.",
    tools=[{
        "type": "code_interpreter"
    }],
    model="gpt-3.5-turbo-1106")

st.button("Start")
if st.button:
  # Generate assistant response based on user input
  thread = client.beta.threads.create()
  message = client.beta.threads.messages.create(
      thread_id=thread.id,
      role="user",
      content=
      f"Help define a {problem_area} {problem_type} problem where we aim to {objectives}."
  )

  run = client.beta.threads.runs.create(thread_id=thread.id,
                                        assistant_id=assistant.id,
                                        instructions=".")

# Click button to display assistant response
st.button("Get response")
if st.button and 'thread' in locals():
  run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
  messages = client.beta.threads.messages.list(thread_id=thread.id)

  # Display message response to user
  if messages:
    st.write(messages)

# # User can upload csv file
# uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# # Display uploaded file
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.write(df)

# # Upload file to openai with assistant purpose
# file = client.files.create(
#   file=open("uploaded_file.csv", "rb"),
#   purpose='assistants'
# )

# thread = client.beta.threads.create(
#   messages=[
#     {
#       "role": "user",
#       "content": "Here are my decisions and constraints",
#       "file_ids": [file.id]
#     }
#   ]
# )
