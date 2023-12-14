import streamlit as st

# Create a placeholder
placeholder = st.empty()

# Now you can write content to the placeholder
placeholder.text('Hello, this text can be replaced!')

# Create an input widget
user_input = st.text_input("Change the text in the placeholder:")

# When the user enters text, update the placeholder with the new text
if user_input:
    placeholder.text(user_input)