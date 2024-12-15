import streamlit as st

from content.intro import show_intro
from content.data_management import show_data_mgmt
from content.visuals import show_graphs
from content.wordcloud import show_wordcloud
from content.investment_app import show_investment_app
from content.conclu import show_conclu

# Dictionary mapping sidebar labels to page functions
pages = {
    "Introduction": show_intro,
    "Data Management": show_data_mgmt,
    "Visualization": show_graphs,
    "Wordcloud": show_wordcloud,
    "Investment app": show_investment_app,
    "Conclusion": show_conclu
}

# Sidebar navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.selectbox("Choisissez une page :", list(pages.keys()))

# Display the selected page
pages[selected_page]()
