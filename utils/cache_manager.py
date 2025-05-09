# utils/cache_manager.py
import streamlit as st
from datetime import datetime
from typing import Any, Optional, Dict
import json
import os
import hashlib

class CacheManager:
    """Gestionnaire centralisé du cache de l'application"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        if 'cache_data' not in st.session_state:
            st.session_state.cache_data = {}
        if 'cache_timestamps' not in st.session_state:
            st.session_state.cache_timestamps = {}
        if 'rate_limit_tracker' not in st.session_state:
            st.session_state.rate_limit_tracker = {}
    
    def _get_cache_key(self, key: str, params: Dict = None) -> str:
        """Génère une clé de cache unique basée sur les paramètres"""
        cache_key = key
        if params:
            params_str = json.dumps(params, sort_keys=True)
            params_hash = hashlib.md5(params_str.encode()).hexdigest()
            cache_key = f"{key}_{params_hash}"
        return cache_key
    
    def get(self, key: str, params: Dict = None, ttl_seconds: int = 3600) -> Optional[Any]:
        """Récupère une valeur du cache si elle est encore valide"""
        cache_key = self._get_cache_key(key, params)
        
        # Vérifier d'abord la session
        if cache_key in st.session_state.cache_data:
            timestamp = st.session_state.cache_timestamps.get(cache_key)
            if timestamp and (datetime.now() - timestamp).total_seconds() < ttl_seconds:
                return st.session_state.cache_data[cache_key]
        
        # Vérifier le cache disque
        file_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    cached = json.load(f)
                timestamp = datetime.fromisoformat(cached['timestamp'])
                if (datetime.now() - timestamp).total_seconds() < ttl_seconds:
                    st.session_state.cache_data[cache_key] = cached['data']
                    st.session_state.cache_timestamps[cache_key] = timestamp
                    return cached['data']
            except:
                pass
        
        return None
    
    def set(self, key: str, value: Any, params: Dict = None):
        """Stocke une valeur dans le cache"""
        cache_key = self._get_cache_key(key, params)
        
        # Stocker en session
        st.session_state.cache_data[cache_key] = value
        st.session_state.cache_timestamps[cache_key] = datetime.now()
        
        # Stocker sur disque pour persistance
        file_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(file_path, 'w') as f:
                json.dump({
                    'data': value,
                    'timestamp': datetime.now().isoformat()
                }, f)
        except:
            pass  # Ignorer les erreurs de sérialisation
    
    def clear(self, key: str = None, params: Dict = None):
        """Efface le cache pour une clé spécifique ou tout le cache"""
        if key:
            cache_key = self._get_cache_key(key, params)
            if cache_key in st.session_state.cache_data:
                del st.session_state.cache_data[cache_key]
            if cache_key in st.session_state.cache_timestamps:
                del st.session_state.cache_timestamps[cache_key]
            
            file_path = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
        else:
            st.session_state.cache_data = {}
            st.session_state.cache_timestamps = {}
            for file in os.listdir(self.cache_dir):
                if file.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, file))
    
    def check_rate_limit(self, key: str, limit_per_minute: int = 60) -> bool:
        """Vérifie si on peut faire un appel en respectant le rate limit"""
        now = datetime.now()
        minute_key = now.strftime("%Y-%m-%d_%H:%M")
        
        if key not in st.session_state.rate_limit_tracker:
            st.session_state.rate_limit_tracker[key] = {}
        
        tracker = st.session_state.rate_limit_tracker[key]
        
        # Nettoyer les anciennes entrées
        old_keys = [k for k in tracker.keys() if k != minute_key]
        for k in old_keys:
            del tracker[k]
        
        # Vérifier le compteur actuel
        current_count = tracker.get(minute_key, 0)
        if current_count >= limit_per_minute:
            return False
        
        # Incrémenter le compteur
        tracker[minute_key] = current_count + 1
        return True
    
    def get_or_compute(self, key: str, compute_func, params: Dict = None, ttl_seconds: int = 3600):
        """Récupère du cache ou calcule la valeur"""
        cached_value = self.get(key, params, ttl_seconds)
        if cached_value is not None:
            return cached_value
        
        value = compute_func()
        self.set(key, value, params)
        return value