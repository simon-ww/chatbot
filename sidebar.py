import streamlit as st

def render_sidebar():
    """Render the sidebar and return the selected navigation option."""
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    menu_options = ["Personality Analysis", "AI Coach", "Negotiation Advice"]
    choice = st.sidebar.radio("Go to", menu_options)

    # Sidebar Settings
    st.sidebar.title("Settings")
    dark_mode = st.sidebar.checkbox("Enable Dark Mode", value=False)

    # Apply Dark Mode
    if dark_mode:
        st.markdown(
            """
            <style>
            body {
                background-color: #0e1117;
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    return choice
