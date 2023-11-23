import streamlit as st
import openai
# import pymoo as pymoo
import os

st.title("Optimization Assistant")
st.write(
    "This is an optimization assistant that helps domain experts define, select, and interpret multi-objective optimization problems."
)

st.header("Step 1: Define the problem")

# Initialize the OpenAI client
client = openai.OpenAI()

# Setup OpenAI assistant to guide user through optimization problem
# assistant = client.beta.assistants.create(
#     name="Optimization Assistant",
#     instructions=
#     "You are an expert at leveraging multi-objective optimization to improve planning and operations of complex networks. You will help domain experts define problems, select methods, run, and interpret results.",
#     tools=[{
#         "type": "code_interpreter"
#     }],
#     model="gpt-3.5-turbo-1106")
assistant_id = "asst_60dvjUYk1TZJuNgVFp48bKAO"


# Function to display problem definition widgets
def define_problem():
  problem_area = st.selectbox(
      "Select a problem area:",
      ["Energy", "Water", "Transportation", "Manufacturing", "Supply Chain"],
      key='problem_area')

  problem_type = st.selectbox("Select a problem type:",
                              ["Strategic Planning", "Operations"],
                              key='problem_type')

  objectives = st.multiselect(
      "Select optimization objectives:",
      ["Minimize Cost", "Maximize Benefits", "Minimize Risk"],
      key='objectives')
  return problem_area, problem_type, objectives


# Start session button
if st.button("Start Session"):
  # Create a thread and store in session state
  # Uncomment the following line and comment out the placeholder when integrating with OpenAI
  # thread = client.beta.threads.create()
  # st.session_state['thread_id'] = thread.id
  st.session_state[
      'thread_id'] = "thread_LkRzVY7uFm3OyjV9NYbVqQ2u"  # Placeholder thread ID

# Display the problem definition menus and get the values
problem_area, problem_type, objectives = define_problem()

# Define problem and get response button
if st.button("Define Problem and Get Response"):
  if 'thread_id' in st.session_state:
    # Send message to thread
    message = client.beta.threads.messages.create(
        thread_id=st.session_state['thread_id'],
        role="user",
        content=
        f"Help define a {problem_area} {problem_type} problem where we aim to {', '.join(objectives)}."
    )

    # Create and execute a run
    run = client.beta.threads.runs.create(
        thread_id=st.session_state['thread_id'],
        assistant_id=assistant_id,
        instructions=
        "The user is a domain expert. Please help them define and solve their optimization problem."
    )

    # Store run ID in session state
    st.session_state['run_id'] = run.id

# # Display the problem definition menus regardless of button states
# define_problem()

# Display response button
if st.button("Display Response"):
  if 'thread_id' in st.session_state and 'run_id' in st.session_state:
    # Retrieve the run to get the response
    run = client.beta.threads.runs.retrieve(
        thread_id=st.session_state['thread_id'],
        run_id=st.session_state['run_id'])

    # List messages in the thread
    messages = client.beta.threads.messages.list(
        thread_id=st.session_state['thread_id'])

    # Check if messages are present
    if messages:
      # Filter out only assistant's messages
      assistant_messages = [msg for msg in messages if msg.role == 'assistant']

      # Get the latest assistant message
      if assistant_messages:
        latest_message = assistant_messages[0]  # Last message in the list
        message_content = latest_message.content if latest_message.content else 'No content available'

        # Display the message content
        st.write("Assistant's Response:")
        st.write(message_content)
      else:
        st.write("No responses from the assistant found.")
    else:
      st.write("No messages found in the thread.")

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
