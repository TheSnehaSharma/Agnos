import streamlit as st
import pandas as pd
import json
import os
import base64
import time
import hashlib
import uuid
from datetime import datetime

# --- CONFIGURATION ---
DATA_DIR = "agnos_data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
AUTH_FILE = os.path.join(DATA_DIR, "auth_registry.json")

# --- ROBUST ASSET LOADER ---
def load_asset(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "assets", filename)
    try:
        with open(file_path, "r") as f: return f.read()
    except FileNotFoundError:
        st.error(f"❌ Critical: '{filename}' not found in assets folder.")
        st.stop()
        return ""

st.set_page_config(page_title="Agnos Enterprise", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{load_asset('style.css')}</style>", unsafe_allow_html=True)

# --- BACKEND HELPERS ---
def load_auth():
    if os.path.exists(AUTH_FILE):
        with open(AUTH_FILE, "r") as f: return json.load(f)
    return {}

def save_auth(data):
    with open(AUTH_FILE, "w") as f: json.dump(data, f)

def get_file_paths(org_key):
    return {
        "db": os.path.join(DATA_DIR, f"faces_{org_key}.json"),
        "logs": os.path.join(DATA_DIR, f"logs_{org_key}.csv")
    }

# --- SESSION STATE ---
if "auth_status" not in st.session_state: st.session_state.auth_status = False
if "org_key" not in st.session_state: st.session_state.org_key = None
if "db" not in st.
