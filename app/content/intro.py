import pandas as pd
import streamlit as st



def show_intro():
    #st.set_page_config(layout="wide")

    st.header("**Data Management, Data Viz & Text Mining**")
    st.subheader('''SDA session 2024-2025''')
    st.markdown(''':gray[COTARTA Anca-Madalina, JABEUR Yasmine, PERAUDEAU Paul-Elie, SAHIB Nassim]''')
    
    ###########
    # Texte de l'introduction
    ###########

    st.subheader("*Introduction*")
    st.markdown("Cette application streamlit, réalisée en groupe de 4, a pour objectif de répondre au sujet suivant:")
    st.markdown("Réalisation d’une application Streamlit qui permet d’afficher tout ce qu’il y a à savoir sur votre jeu de données. L’application devra contenir les informations suivantes :")
    st.markdown("*● Description du jeu de données : l’origine des données, nombre d’observations, nombre de variables, types de variables, signification de chaque variable, nombre de valeurs manquantes par variable*")
    st.markdown("*● Statistiques descriptives*")
    st.markdown("*● Visualisations : Au moins cinq graphiques avec quelques filtres interactifs.*")

    ###########
    # Texte custom pour notre organisation
    ###########

    st.subheader("*Organisation du projet*")
    st.markdown("*Afin de répondre aux questions, nous avons divisé notre travail de la manière suivante*")
    st.markdown("1. Tout d'abord un notebook, présent dans les fichiers du projet, mais également dans la page nommée :blue[*data management*]. Cette page récapitule nos observations du dataset et ses transformations pour le rendre exploitable.")
    st.markdown("2. En second, grâce au travail réalisé dans le notebook, nous avons créé plusieurs visuels et widget permettant de comprendre dynamiquement les données dans la page :blue[*Visualization*], .")
    st.markdown("3. En suivant, nous avons choisi d'afficher le wordcloud dans une page dédiée :blue[*Wordcloud*].")
    st.markdown("4. Pour finir le projet nous nous sommes accordé sur la création d'une petite application d'investissement, permettant de comparer des stratégies et leurs retours par rapport à un benchmark (le S&P 500). Nous vous laissons la découvrir dans la page :blue[*Investment app*]")
    st.markdown("Tout le projet est divisé à travers le dossier de l'application, afin de rendre la rendre la plus facile à debugger et améliorer. L'application principale, nommée :blue[*app.py*] ne sert donc qu'à appeler les pages définies dans le dossier :blue[*content*]. Les fonctions des pages sont quand à elles définies dans le dossier :blue[*utils*].")