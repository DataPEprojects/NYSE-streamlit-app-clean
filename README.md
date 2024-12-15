# NYSE-streamlit-app
## Lien de l'application
https://nyse-app-app-clean-zjxzjxxkxfev7jnmriuypg.streamlit.app/

## Présentation du Projet
Ce projet a pour objectif de fournir une application Streamlit permettant une analyse complète d’un jeu de données. Réalisée par un groupe de quatre étudiants, l'application offre diverses fonctionnalités allant de la gestion et la visualisation des données jusqu’à la création d'un wordcloud, en passant par une application d'investissement comparative.

## Fonctionnalités Clés

Description du Jeu de Données
Origine des données
Dimensions du dataset (nombre d’observations, nombre de variables)
Types et significations des variables
Nombre de valeurs manquantes par variable
Statistiques Descriptives
Calculs et présentations de statistiques clés (moyenne, médiane, écart-type, etc.)

## Visualisations Interactives

Au moins cinq graphiques interactifs
Des filtres dynamiques permettant d’explorer les données selon différents critères

## Wordcloud

Génération d’un nuage de mots permettant de visualiser rapidement les termes les plus fréquents dans les données textuelles.

## Application d’Investissement (Investment App)

Comparaison de stratégies d’investissement basées sur le jeu de données
Benchmark avec le S&P 500
Analyse des performances cumulées, volatilité, drawdown, Sharpe ratio, etc.

## Organisation du Projet
Le projet est structuré autour d’une application principale Streamlit (app.py) et de différentes pages organisées dans des répertoires dédiés :

data management : Cette partie, présentée dans un notebook et accessible également via la page Data Management, décrit l’observation initiale du dataset, les transformations et nettoyages nécessaires, ainsi que la préparation des données pour une analyse ultérieure.

visualization : Cette section, accessible depuis la page Visualization, propose des graphiques interactifs et des widgets permettant une compréhension dynamique et visuelle des données.

wordcloud : La page Wordcloud génère un nuage de mots, facilitant ainsi l’analyse rapide des données textuelles.

investment app : La page Investment app propose une petite application d’investissement. Elle permet de tester différentes stratégies, de suivre leur évolution dans le temps et de les comparer par rapport à un indice de référence (le S&P 500).

La logique du projet est distribuée entre plusieurs dossiers afin d’en faciliter la maintenance et l’amélioration. Le fichier app.py sert de point d’entrée et se contente d’appeler les différentes pages définies dans le dossier content. Les fonctions et la logique métiers de chaque page sont implémentées dans le dossier utils.

# Prérequis
Python 3.7 ou plus récent
Packages Python requis (installables via pip install -r requirements.txt, si un fichier de dépendances est disponible)
Streamlit installé
Lancement de l’Application
Pour lancer l’application, exécutez la commande suivante dans votre terminal, depuis la racine du projet :

streamlit run app.py
Cela ouvrira l’application dans votre navigateur par défaut. Vous pourrez alors naviguer entre les différentes pages (Data Management, Visualization, Wordcloud, Investment App) grâce à la barre latérale fournie par Streamlit.

# Auteurs
COTARTA Anca-Madalina
JABEUR Yasmine
PERAUDEAU Paul-Elie
SAHIB Nassim
