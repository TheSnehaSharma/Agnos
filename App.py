import logging

import streamlit as st

st.set_page_config(page_title="Agnos", page_icon="👁️", layout="wide")

import ui

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

ui.run()
