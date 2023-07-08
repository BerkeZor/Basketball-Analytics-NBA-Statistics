import streamlit as st

st.set_page_config(
    page_title="NBA Stats & Basketball Analytics",
    page_icon="🏀",
)


st.sidebar.success("Select Nba All Star Predictor to find out if the player would become an All-Star")
st.sidebar.success("Select Players Correlation to learn about Players Stats and Correlation Heatmap")
st.image("nbastatslogo.png")
st.write("# Welcome to NBA Stats & Basketball Analytics 👋")
st.markdown(
    """NBA STATS & BASKETBALL ANALYTICS is an app which is provide statistics about NBA Players
    """
)

st.metric(label="How Many Players Are In The NBA", value=450, delta=15,
    delta_color="off")

st.metric(label="How Many Teams Are In The NBA", value=30, delta=0,
    delta_color="off")

