import streamlit as st
import openai

# Set up your OpenAI API key
openai.api_key = "your-openai-api-key"

def get_openai_response(prompt):
    """Function to get a response from OpenAI."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Error: {e}"

# Streamlit app
st.title("OpenAI Chatbot")

# Input from the user
user_input = st.text_input("You: ", placeholder="Type your message here...")

# Display conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Button to send the message
if st.button("Send"):
    if user_input:
        # Append user input to conversation history
        st.session_state.conversation.append({"role": "user", "content": user_input})
        
        # Get OpenAI response
        bot_response = get_openai_response(user_input)
        st.session_state.conversation.append({"role": "assistant", "content": bot_response})

# Display the conversation history
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Bot:** {message['content']}")
