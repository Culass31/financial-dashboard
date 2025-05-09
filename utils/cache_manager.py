# utils/cache_manager.py
import streamlit as st
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
import json
import os

class CacheManager:
    """Gestionnaire centralisé du cache de l'application"""
    
    def __init__(self):
        if 'cache_data' not in st.session_state:
            st.session_state.cache_data = {}
        if 'cache_timestamps' not in st.session_state:
            st.session_state.cache