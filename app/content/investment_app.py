import streamlit as st
from datetime import datetime
import plotly.express as px
from utils.functions.functions import (load_data, load_sp500_data, compute_portfolio_performance, create_comparison_df)

st.set_page_config(layout="wide")

def show_investment_app():
    st.title(":blue[Investment App - Analyse des Performances (Portefeuille vs S&P 500)]")

    # Load data
    try:
        df = load_data()
        sp500_data = load_sp500_data(df)
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        st.stop()

    # Descriptive text before widgets
    try:
        st.markdown("Afin de pousser plus loin notre projet, nous avons choisi de réaliser une tâche similaire à celles que l'on peut observer dans notre environnement de travail, la finance de marché.")
        st.markdown("Cette application permet de simuler un investissement sur le :blue[NYSE], et de le comparer avec le benchmark classique des marchés américains : le :blue[S&P 500]. Nous avons choisi le S&P car il n'existe pas vraiment d'indices connus sur le NySE, mais le S&P reprends beaucoup de ses valeurs.")
        st.markdown("Afin de vous servir de l'application, :blue[sélectionnez une ou bien plusieurs entreprises], ainsi qu'une :blue[somme de départ et une plage temporelle]. Votre argent de départ sera réparti de manière égale dans toutes les entreprises.")
        
        min_date = df['date'].min()
        max_date = df['date'].max()

        min_date_dt = datetime(min_date.year, min_date.month, min_date.day)
        max_date_dt = datetime(max_date.year, max_date.month, max_date.day)

        col1, col2, col3 = st.columns(3)
        with col1:
            start_time, end_time = st.slider(
                "Sélectionnez une plage de dates :",
                min_value=min_date_dt,
                max_value=max_date_dt,
                value=(min_date_dt, max_date_dt),
                format="DD/MM/YY"
            )
        with col2:
            starting_amount = st.number_input(
                "Entrez le montant de départ (€)", min_value=1, max_value=9999999, value=1000, step=100
            )
        with col3:
            unique_companies = df['Security'].unique()
            selected_companies = st.multiselect(
                "Sélectionnez une ou plusieurs entreprises :",
                options=unique_companies,
                default=[]
            )

    except Exception as e:
        st.error(f"Erreur lors de la création des widgets : {e}")
        st.stop()

    # Compute and show performance
    if st.button("Afficher la performance cumulée"):
        st.write("Calculs en cours...")
        results = compute_portfolio_performance(df, selected_companies, start_time, end_time, starting_amount)
        
        if results is None:
            if not selected_companies:
                st.warning("Veuillez sélectionner au moins une entreprise.")
            else:
                st.warning("Aucune donnée pour ces critères.")
            return
        
        portfolio_performance = results['portfolio_performance']
        max_drawdown = results['max_drawdown']
        vol = results['vol']
        final_value = results['final_value']
        performance_pct = results['performance_pct']
        sharpe_ratio = results['sharpe_ratio']

        # Display metrics
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.write("*Maximum Drawdown du Portefeuille (%) :*", round(max_drawdown, 2))
            st.write("*Volatilité Annualisée du Portefeuille (%) :*", round(vol, 2))

        with result_col2:
            st.write("*Valeur Finale du Portefeuille (€) :*", round(final_value, 2))
            st.write("*Performance Totale du Portefeuille (%) :*", round(performance_pct, 2))
            st.write("*Sharpe Ratio du Portefeuille :*", round(sharpe_ratio, 2))

        # Create and display chart
        portfolio_comparison = create_comparison_df(portfolio_performance, sp500_data, start_time, end_time, starting_amount)
        fig = px.line(
            portfolio_comparison,
            x='date',
            y=['Portefeuille', 'S&P 500'],
            title="Portefeuille Équipondéré vs S&P 500",
            labels={'value': 'Valeur de l\'Investissement (€)', 'variable': 'Performance'}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("Calculs terminés 😊")