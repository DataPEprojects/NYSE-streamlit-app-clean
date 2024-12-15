import pandas as pd
import streamlit as st
from utils.functions.functions import select_security
from utils.graphs.graphs import plot_cumulative_log_returns_by_sector,plot_treemap_top_companies,plot_security_vs_sector,plot_security_histogram,plot_single_state

def show_graphs():
    st.header("Visualization:")
    st.markdown("Veuillez trouver ci-dessous nos propositions de visualisation pour le dataset:")
    st.subheader("Partie 1 : Etude sur une entreprise")

    col1, col2 = st.columns([2, 3])  

    with col1:
        choice = select_security('Security') 
    with col2:
        choice2 = st.radio("Quelle valeur voulez vous afficher ?",
                           ["open","close","low","high","volume"],
                           horizontal=True, )  
        
    plot_security_vs_sector(choice)
    
    col1, col3 = st.columns([3,2])
    with col1:
        plot_single_state(choice)

    with col3:
        plot_security_histogram(choice,y_column=choice2)

        
    st.subheader("Partie 2 : Etudes globales")
    col1, col2 = st.columns([2, 3])


    with col1:
        number = st.number_input(
            "Sélectionnez le nombre d'entreprises à afficher",
            min_value=30,
            max_value=120,
            value=80,
            step=1
        )
    with col2:
        pass
    plot_treemap_top_companies(number)  # Treemap plot
    st.markdown("Ci-dessous la performance des différents secteurs sur une échelle logarithmique")
    plot_cumulative_log_returns_by_sector()