import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

def plot_security_data(selected_security,y_column):
    """
    Plots a linear graph for the selected 'Security' using plotly.

    :param selected_security: The Security chosen by the user.
    :param df: The DataFrame containing the data.
    """
    if selected_security is not None:
        # Filter the DataFrame for the selected security
        df = pd.read_csv('data/final.csv')
        filtered_df = df[df['Security'] == selected_security]

        if filtered_df.empty:
            st.warning(f"Aucune donnée disponible pour {selected_security}.")
            return

        # Create the plotly linear graph
        fig = px.line(
            filtered_df,
            x='date',
            y=y_column,
            title=f"Prix de Clôture pour {selected_security}",
            labels={'date': 'Date', 'close': 'Prix de Clôture (€)'},
            template='plotly_white'
        )

        # Display the graph in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Aucun Security sélectionné ou aucune donnée disponible.")
def plot_treemap_top_companies(number):
    """
    Affiche un graphique en carré représentant la part des volumes par entreprise.
    Le nombre d'entreprises à afficher individuellement est déterminé par 'number'.
    """
    try:
        # Charger les données
        df = pd.read_csv('data/final.csv')
        
        # Calculer les volumes totaux par entreprise
        total_volumes = df.groupby('Security')['volume'].sum().reset_index()
        
        # Trier par volume décroissant
        total_volumes = total_volumes.sort_values(by='volume', ascending=False)

        # Séparer les `number` premières entreprises
        top_companies = total_volumes.iloc[:number].reset_index(drop=True)

        # Ajouter une colonne de couleur pour le treemap
        max_volume = top_companies['volume'].max()
        top_companies['color'] = top_companies['volume']

        # Créer le treemap
        fig = px.treemap(
            top_companies,
            path=['Security'],
            values='volume',
            color='color',
            title="Répartition des volumes totaux par entreprise (Top Entreprises)",
            labels={'volume': 'Volumes totaux'},
            color_continuous_scale=px.colors.sequential.Viridis,
            template='plotly_white'
        )

        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors de la création du graphique : {e}")


def plot_security_vs_sector(selected_security, y_column='normalized_close'):
    """
    Line chart qui compare les perfs du secteur et celles de l'entreprise selectionnée
    """
    df = pd.read_csv('data/final.csv')

    # Filtre pour l'entreprise
    filtered_security = df[df['Security'] == selected_security]
    if filtered_security.empty:
        st.warning(f"Aucune donnée disponible pour {selected_security}.")
        return

    sector = filtered_security['GICS Sector'].iloc[0]
    filtered_sector = df[df['GICS Sector'] == sector]

    # Créer la ligne du secteur en moyenne
    sector_aggregated = (
        filtered_sector.groupby('date')[y_column]
        .mean()
        .reset_index()
        .rename(columns={y_column: 'Sector Average'})
    )

    security_data = filtered_security[['date', y_column]].rename(columns={y_column: selected_security})

    # Fusion des datas pour comparaison
    merged_data = pd.merge(
        security_data,
        sector_aggregated,
        on='date',
        how='inner'
    )

    fig = px.line(
        merged_data,
        x='date',
        y=[selected_security, 'Sector Average'],
        title=f"Comparaison entre {selected_security} et la Moyenne du Secteur ({sector})",
        labels={'date': 'Date', 'value': 'Performance'},
        template='plotly_white'
    )

    fig.update_layout(
        legend=dict(
            orientation="h", 
            yanchor="bottom",
            y=-0.3,  # Eviter que les légendes ne se fondent entre elles
            xanchor="center",
            x=0.5,
            title_text="",
            itemclick="toggleothers", 
            itemdoubleclick="toggle" 
        )
    )


    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

def plot_security_histogram(selected_security, y_column):
    """
    Plots a histogram for the selected security and metric.

    :param selected_security: The selected security.
    :param y_column: The column/metric to visualize.
    """
    # Load data
    df = pd.read_csv('data/final.csv')

    # Filter for the selected security
    filtered_security = df[df['Security'] == selected_security]
    if filtered_security.empty:
        st.warning(f"Aucune donnée disponible pour {selected_security}.")
        return

    # Create the histogram
    fig = px.histogram(
        filtered_security,
        x=y_column,
        nbins=20,
        title=f"Distribution de {y_column} pour {selected_security}",
        labels={y_column: y_column.capitalize()},
        template='plotly_white'
    )

    # Customize layout for readability
    fig.update_layout(
        xaxis_title=y_column.capitalize(),
        yaxis_title='Fréquence',
        showlegend=False
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)


# https://gist.github.com/JeffPaine/3083347 nous avons trouvé cette liste à l'adresse suivante, elle nous sert à montrer dans quel état se situe l'entreprise

STATE_ABBREVIATIONS = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
}

# cette fonction est l'une des plus ambitieuse du projet, pour information la couleur transparent est rgba(0, 0, 0, 0)

def plot_single_state(selected_company):
    """
    Création d'une carte montrant ou se trouve l'entreprise
    """
    # Load dataset
    df = pd.read_csv("data/final.csv")

    # Get the company's state from the address
    company_data = df[df["Security"] == selected_company]
    if company_data.empty:
        st.warning(f"Aucune donnée trouvée pour l'entreprise : {selected_company}.")
        return

    address = company_data.iloc[0]["Address of Headquarters"]

    _, state = address.split(", ")  # On veut unniquement la région, donc on split en deux l'addresse
    state = state.strip()

    # On laisse cette sécurité, je ne sais pas si la présence du QG sur sol USA est obligatoire 
    if state not in STATE_ABBREVIATIONS:
        st.warning(f"L'état '{state}' n'est pas reconnu.")
        return

    state_abbreviation = STATE_ABBREVIATIONS[state]

    # On met 1 de valeur à l'état correspondant seulement
    state_data = pd.DataFrame({
        "State": [state_abbreviation],
        "Highlight": [1]  # Highlight value
    })

    # Create the map
    fig = px.choropleth(
        state_data,
        locations="State",
        locationmode="USA-states",  # Use state abbreviations for the USA map
        color="Highlight",
        scope="usa",
        color_continuous_scale=[[0, "rgba(0, 0, 0, 0)"], [1, "#FF0000"]],
        title=f"Highlighting {state} for {selected_company}",
    )

    # Customize the map layout
    fig.update_geos(
        showframe=False,
        showcoastlines=False,
        projection_type="albers usa",
        visible=True,
        showland=True,
        landcolor="rgba(0, 0, 0, 0)",  # Fully transparent land for all non-highlighted areas
        subunitcolor="white",  # White borders for states
        countrycolor="white",  # White borders for the country
        bgcolor="rgba(0, 0, 0, 0)"  # Transparent surroundings
    )

    # Adjust layout for the legend
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        paper_bgcolor="rgba(0, 0, 0, 0)",  # Fully transparent background
        geo_bgcolor="rgba(0, 0, 0, 0)",  # Transparent background
        coloraxis_showscale=False  # Remove the color scale bar
    )

    # Show the map
    st.plotly_chart(fig, use_container_width=True)

def plot_cumulative_log_returns_by_sector():
    """
    Le graphique déjà présent dans le notebook des différents secteurs en log normal
    """
    df = pd.read_csv("data/final.csv")

    # Calcul des rendements logarithmiques
    df['log_return'] = df.groupby('symbol')['close'].transform(lambda x: np.log(x / x.shift(1)))

    # Moyenne des rendements logarithmiques par secteur pour chaque jour
    sector_log_returns = df.groupby(['date', 'GICS Sector'])['log_return'].mean().reset_index()

    # Calcul des rendements cumulés logarithmiques par secteur
    sector_log_returns['cumulative_log_return'] = sector_log_returns.groupby('GICS Sector')['log_return'].cumsum()

    # Créer le graphique
    fig = px.line(
        sector_log_returns,
        x="date",
        y="cumulative_log_return",
        color="GICS Sector",
        title="Rendements Cumulés Logarithmiques par Secteur",
        labels={
            "date": "Date",
            "cumulative_log_return": "Rendements Cumulés Logarithmiques",
            "GICS Sector": "Secteur GICS"
        },
        template="plotly_white"
    )

    # Ajuster la légende
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig, use_container_width=True)
