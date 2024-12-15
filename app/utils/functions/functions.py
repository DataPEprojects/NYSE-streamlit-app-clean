import nbformat
from nbclient import NotebookClient

# Streamlit
import streamlit as st

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.io as pio

# Data Handling
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

import yfinance as yf
from datetime import datetime

########################
# FOR finance 
########################

def flatten_columns(df):
    """Flatten multi-level columns in case they exist."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(str(c) for c in col if c != '') for col in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    return df

def load_data():
    """Load and preprocess the main dataset."""
    df = pd.read_csv('data/final.csv')
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df.sort_values(by=['symbol', 'date']).reset_index(drop=True)
    df = flatten_columns(df)
    df['log_return'] = df.groupby('symbol')['close'].transform(lambda x: np.log(x / x.shift(1)))
    df['cumulative_log_return'] = df.groupby('symbol')['log_return'].cumsum().fillna(0)
    df = flatten_columns(df)
    return df

def load_sp500_data(df):
    """Load and preprocess the S&P 500 data based on the date range of df."""
    start_date = df['date'].min()
    end_date = df['date'].max()
    sp500_data = yf.download("^GSPC", start=start_date, end=end_date)
    sp500_data.reset_index(inplace=True)
    sp500_data = sp500_data[['Date', 'Adj Close']].rename(columns={'Date': 'date', 'Adj Close': 'S&P 500'})
    sp500_data['cumulative_return'] = np.log(sp500_data['S&P 500'] / sp500_data['S&P 500'].iloc[0])
    sp500_data = flatten_columns(sp500_data)
    return sp500_data

def compute_portfolio_performance(df, selected_companies, start_time, end_time, starting_amount):
    """Compute the portfolio performance metrics and return the results as a dictionary."""
    filtered_df = df[(df['date'] >= start_time) & (df['date'] <= end_time)]
    if selected_companies:
        filtered_df = filtered_df[filtered_df['Security'].isin(selected_companies)]
    else:
        return None  # Handle in calling code

    if filtered_df.empty:
        return None

    num_companies = len(selected_companies)
    allocated_amount = round(starting_amount / num_companies, 1)
    filtered_df['Investment_Value'] = allocated_amount * np.exp(filtered_df['cumulative_log_return'])

    portfolio_performance = filtered_df.groupby('date')['Investment_Value'].sum().reset_index()
    portfolio_performance = flatten_columns(portfolio_performance)

    portfolio_performance['Daily Return'] = portfolio_performance['Investment_Value'].pct_change()
    portfolio_performance['Cumulative Max'] = portfolio_performance['Investment_Value'].cummax()
    portfolio_performance['Drawdown'] = (portfolio_performance['Investment_Value'] / portfolio_performance['Cumulative Max'] - 1) * 100

    max_drawdown = portfolio_performance['Drawdown'].min()
    vol = portfolio_performance['Daily Return'].std() * np.sqrt(252) * 100
    final_value = portfolio_performance['Investment_Value'].iloc[-1]
    performance_pct = (final_value / starting_amount - 1) * 100
    risk_free_rate = 0.02
    annual_return = portfolio_performance['Daily Return'].mean() * 252
    sharpe_ratio = (annual_return - risk_free_rate) / (portfolio_performance['Daily Return'].std() * np.sqrt(252))

    return {
        'portfolio_performance': portfolio_performance,
        'max_drawdown': max_drawdown,
        'vol': vol,
        'final_value': final_value,
        'performance_pct': performance_pct,
        'sharpe_ratio': sharpe_ratio
}

def create_comparison_df(portfolio_performance, sp500_data, start_time, end_time, starting_amount):
    """Create a DataFrame for portfolio vs. S&P 500 comparison."""
    sp500_filtered = sp500_data[(sp500_data['date'] >= start_time) & (sp500_data['date'] <= end_time)].copy()
    sp500_filtered['SP500_Investment'] = starting_amount * np.exp(sp500_filtered['cumulative_return'])
    sp500_filtered = flatten_columns(sp500_filtered)

    portfolio_comparison = pd.merge(
        portfolio_performance[['date','Investment_Value']], 
        sp500_filtered[['date','SP500_Investment']],
        on='date', how='inner'
    )
    portfolio_comparison = flatten_columns(portfolio_comparison)
    portfolio_comparison.rename(columns={'Investment_Value':'Portefeuille', 'SP500_Investment':'S&P 500'}, inplace=True)

    return portfolio_comparison


########################
# FOR VISUALS
########################


def select_security(parameter):
    """
    Creates a selectbox widget for unique 'Security' values in 'data/final.csv'
    and returns the user's selection.
    """
    try:
        # Load the CSV file
        df = pd.read_csv('data/final.csv')
        
        # Get unique 'Security' values
        unique_securities = df[parameter].unique()

        # Create a selectbox widget
        selected_security = st.selectbox(
            "Sélectionnez une entreprise (Security):",
            options=unique_securities,
            index=0  # Default selection is the first one
        )
        return selected_security

    except Exception as e:
        st.error(f"Erreur lors du chargement des données ou de la création du widget : {e}")
        return None



def display_wordcloud(file_path):
    """
    Faire appraître le Jupyter Notebook dans l'app Streamlit.
    
    :param file_path: le path vers le Jupyter Notebook file (.ipynb)
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        notebook = nbformat.read(file, as_version=4)

    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            st.markdown(cell['source'])
        elif cell['cell_type'] == 'code':
            st.code(cell['source'])

            '''
            Le soucis était que les mots n'apparaissaient pas, nous avons du rajouter cette fonction manuellement pour faire apparaître le graphique
            '''
            if 'words_stemmed' in cell['source'] and 'WordCloud' in cell['source']:
                # Mock `corpus_stemmed` for demonstration purposes
                corpus_stemmed = ['march taux dinteret lanne anne rest marque dun pierr blanch taux dinteret serv oblig tomb massiv terrain negat even lhistoir ampleur mont oblig serv taux negat depass fin danne 10000 milliard dollar mond atteint 13000 milliard cour danne dautr term acheteur oblig lieu percevoir remuner form taux dinteret contrair perd sil conservent titr jusqu dat rembours rend pari poursuit hauss cour oblig permettr revendr titr plusvalu arrive echeanc strateg risque entrain lourd pert cas retourn baiss cour oblig situat anormal provoque polit ultraaccommod banqu central cris financier mondial anne taux dinteret mondial nont cess chut cris accompagn polit taux zero banqu central cot cour oblig evoluent linvers taux dinteret flamb leffet dun fort demand dactif consider valeursrefug oblig emis etat solid etatsun allemagn franc juge risque invest action anne renforc bce polit dassoupl quantit augment pression taux 10 mar banqu central europeen abaiss taux depot 04 taux refinanc 0 bce port 60 80 milliard deuros mont achat dactif march financi jusquen mar anne mont achat obligatair ensuit reduit 60 mdse programm prolong jusqu fin anne juin vot britann faveur brex acceler lafflux capital oblig envoi taux niveau histor semain referendum britann juillet taux lemprunt detat allemand bund 10 an tomb temp 015 tand taux japon 10 an atteint 029 rend loat francais 10 an chut 009 tbond americain 10 an 136 fac situat extrem critiqu multiplie attir lattent risqu elev cour moyen term taux negat system financi mondial vu taux voir negat bon nouvel leconom soutenu march immobili franc taux emprunt hypothecair atteint histor anne permet menag pouvoir acced propriet taux favorisent lacc cred entrepris natur relanc invest lemploi sil cadeau acteur econom sendettent taux negat revanch maledict revenus deriv revenus plac financi premier victim taux banqu meti consist emprunt taux pret taux elev chut taux reduct lecart taux long court pesent benefic etabl bancair banqu allemagn commenc factur client compt depot baiss taux lamin rend produit depargn livret assurancev fond retrait met dang vers futur retrait pay disposent fond pension juin meilleur special mondial march obligatair bill gross tir sonnet dalarm format dun bull obligatair linvestisseur americain fondateur celebr fond obligatair pimco patron fond janus capital soulign taux dinteret mondial tomb 500 an dhistor connu bill gross compar march obligatair mondial supernov explos jour caus gros degat taux mettront remont remonte taux dailleur profil fin lanne anne doubl impact lelect surpris donald trump president etatsun 8 novembr suiv hauss taux directeur fed decide 14 decembr even don signal dun rebond taux etatsun sest repercut mond enti 30 decembr taux 10 an remont allemagn 020 franc 068 japon 004 etatsun 245 taux proch histor taux souverain allemand japon negat jusqu lecheanc 8 an inclus tand franc rend negat jusqu echeanc 6 an etatsun rend revanch posit courb taux 041 mois 30 decembr taux 10 an remont allemagn 020 franc 068 japon 004 etatsun 245 taux proch histor taux souverain allemand japon negat jusqu lecheanc 8 an inclus tand franc rend negat jusqu echeanc 6 an etatsun rend revanch posit courb taux 041 mois 245 10 an 305 30 an europ taux italien portug connu anne hauss anne anne cris det zon euro linstabilit polit faibless secteur bancair ital contribu remonte cout financ plupart expert estiment chut taux enraye sil nanticipent remonte rapid frein croissanc econom europ bce devr poursuivr polit accommod tand fed americain devr continu relev graduel taux cas ralent inattendu leconom americain mondial march taux devr montr volatil prochain mois mesur devoil projet polit econom commercial president donald trump prendr fonction 20 janvi prochain annonc campagn trump relanc budgetair baiss dimpot reglement protection natur cre environ dinflat eleve entretiendr taux dinteret elev quun dollar fort trump prevoit baiss impot particuli entrepris augment fort invest public defens grand traval dinfrastructur programm devr dun part stimul consomm attis linflat dautr part entrain accroissement lendet letat afflux dem obligatair cle environ hauss loffr natur fair baiss cour remont mecan taux anne fed pilot vu cot fed laiss entendr 14 decembr sappret relev fois taux fed fund cour anne port fourchet 125 150 an realis calendri dependr facteur fed maitris commenc nouvel polit econom ladministr trump rappelon quen anne fed initial prevu relev taux fois raison facteur risqu emaill lanne turbulent boursier craint chin brex final agi quun fois decembr an entam cycl haussi hauss dollar evolu haut niveau 13 an fac pani devis mondial obstacl futur action fed export croissanc econom americain souffr effet dun dollar vigour period prolonge']


                # Calculate word frequencies
                words_stemmed = " ".join(corpus_stemmed).split()
                word_counts = Counter(words_stemmed)
                word_freq_dict = dict(word_counts)

                # Generate the Word Cloud
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='viridis',
                    max_words=50
                ).generate_from_frequencies(word_freq_dict)

                # Display the Word Cloud in Streamlit
                st.subheader("Nuage de mots (avec stemmatisation)")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            '''
            Le même problème que pour le wordcloud, nous devons donc demander manuellement de plot le graph
            '''

            if 'bigram_vectorizer' in cell['source']:

                # Generate bi-grams using CountVectorizer
                bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
                X_bigram = bigram_vectorizer.fit_transform(corpus_stemmed)

                # Extract bi-gram frequencies
                bigram_counts = X_bigram.toarray().sum(axis=0)
                bigram_freq = dict(zip(bigram_vectorizer.get_feature_names_out(), bigram_counts))

                # Sort and keep the top 10 bi-grams
                sorted_bigrams = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                bigram_labels, bigram_values = zip(*sorted_bigrams)

                # Create a DataFrame for visualization
                bigram_df = pd.DataFrame({
                    "Bi-grammes": bigram_labels,
                    "Fréquence": bigram_values
                })

                # Generate a horizontal bar chart using Plotly
                fig = px.bar(
                    bigram_df,
                    x="Fréquence",
                    y="Bi-grammes",
                    orientation="h",
                    title="Top 10 des Bi-grammes",
                    labels={"Bi-grammes": "Bi-grammes", "Fréquence": "Fréquence"},
                    #color_discrete_sequence=["#1f77b4"]
                )

                fig.update_layout(
                    plot_bgcolor="black",
                    paper_bgcolor="black",
                    font=dict(size=12),
                    xaxis=dict(title="Fréquence", showgrid=True, gridcolor="gray"),
                    yaxis=dict(title="Bi-grammes", showgrid=False),
                    title=dict(font=dict(size=16))
                )

                # Display the bar chart in Streamlit
                st.plotly_chart(fig)

                # Display the top 10 bi-grams as text
                st.subheader("Top 10 des Bi-grammes")
                for bigram, freq in sorted_bigrams:
                    st.text(f"{bigram}: {freq}")


            # Display the output of code cells
            if 'outputs' in cell:
                for output in cell['outputs']:
                    if output['output_type'] == 'stream':  # Standard output
                        st.text(output['text'])
                    elif output['output_type'] == 'execute_result':  # Results
                        st.text(output['data']['text/plain'])
                    elif output['output_type'] == 'error':  # Errors
                        st.error('\n'.join(output['traceback']))


def display_data_management(file_path):
    """
    Afficher le contenu du notebook et ses graphiques.
    """
    # Charger les données
    final_prices = pd.read_csv("data/final.csv")
    final_prices['date'] = pd.to_datetime(final_prices['date'])

    # Charger le notebook
    with open(file_path, 'r', encoding='utf-8') as file:
        notebook = nbformat.read(file, as_version=4)

    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            st.markdown(cell['source'])
        elif cell['cell_type'] == 'code':
            # Remplacer les chemins dynamiques
            if "BASE_DIR" in cell['source'] and "data_path" in cell['source']:
                cell['source'] = cell['source'].replace(
                    "BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), \"..\"))",
                    "BASE_DIR = os.getcwd()"
                ).replace(
                    "data_path = os.path.join(BASE_DIR, \"data\", \"prices.csv\")",
                    "data_path = \"data/final.csv\""
                )

            # Afficher le code
            st.code(cell['source'])

            # Exécuter le code
            try:
                exec(cell['source'], globals(), locals())
            except Exception as e:
                st.error(f"Erreur lors de l'exécution de la cellule : {e}")

            # Insérer des graphiques spécifiques
            if "sector_indices = final_prices.groupby" in cell['source']:
                st.subheader("Performance (normalisée) par secteur")
                sector_indices = final_prices.groupby(['date', 'GICS Sector'])['normalized_close'].mean().reset_index()
                plt.figure(figsize=(12, 6))
                sns.lineplot(data=sector_indices, x='date', y='normalized_close', hue='GICS Sector', palette="tab10")
                plt.title('Performance (normalisée) par secteur')
                st.pyplot(plt.gcf())
                plt.clf()

            elif "telecom_data = final_prices[final_prices['GICS Sector']" in cell['source']:
                st.subheader("Rendements cumulés des entreprises Télécom")
                telecom_data = final_prices[final_prices['GICS Sector'] == 'Telecommunications Services']
                telecom_data['daily_return'] = telecom_data.groupby('symbol')['close'].pct_change() * 100
                telecom_data['cumulative_return'] = telecom_data.groupby('symbol')['daily_return'].cumsum()
                plt.figure(figsize=(12, 6))
                sns.lineplot(data=telecom_data, x='date', y='cumulative_return', hue='symbol', palette="tab10")
                plt.title('Rendements cumulés des entreprises Télécom')
                st.pyplot(plt.gcf())
                plt.clf()

            elif "sector_log_returns = final_prices.groupby" in cell['source']:
                st.subheader("Rendements cumulés logarithmiques par secteur")
                final_prices['log_return'] = final_prices.groupby('symbol')['close'].transform(lambda x: np.log(x / x.shift(1)))
                sector_log_returns = final_prices.groupby(['date', 'GICS Sector'])['log_return'].mean().reset_index()
                sector_log_returns['cumulative_log_return'] = sector_log_returns.groupby('GICS Sector')['log_return'].cumsum()

                # Graphique pour les rendements cumulés logarithmiques
                plt.figure(figsize=(12, 6))
                sns.lineplot(data=sector_log_returns, x='date', y='cumulative_log_return', hue='GICS Sector', palette="tab10")
                plt.title('Rendements cumulés logarithmiques par secteur', fontsize=16)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Rendements cumulés logarithmiques', fontsize=12)
                plt.legend(loc='upper left', title='Secteurs')
                plt.grid()
                st.pyplot(plt.gcf())
                plt.clf()