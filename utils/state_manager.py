class StateManager:
    """Gestionnaire centralisé de l'état de l'application"""
    
    @staticmethod
    def get_selected_stock():
        return st.session_state.get('selected_stock', None)
    
    @staticmethod
    def set_selected_stock(stock):
        st.session_state['selected_stock'] = stock
    
    @staticmethod
    def get_portfolio():
        if 'portfolio' not in st.session_state:
            st.session_state['portfolio'] = pd.DataFrame()
        return st.session_state['portfolio']