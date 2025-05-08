# Dashboard d'Analyse des Marchés Financiers

Une application Streamlit complète pour l'analyse technique et fondamentale des actions, intégrant les principes d'investissement de Warren Buffett et Benjamin Graham.

## 🚀 Fonctionnalités

### 1. Analyse Technique
- Graphiques interactifs avec Plotly
- Indicateurs techniques (tendance, momentum, Bollinger)
- Analyse de régression
- Recommandations techniques automatisées

### 2. Analyse Fondamentale
- Valorisation intrinsèque multi-méthodes (DCF, Graham, PER, patrimoniale)
- Analyse de l'avantage compétitif
- Analyse des dividendes
- Comparaison avec le consensus des analystes

### 3. Actualités
- Récupération automatique des dernières actualités
- Classification par importance et impact
- Filtrage et tri personnalisés

### 4. Screener Buffett-Graham
- Filtrage multi-critères des actions
- Score composite basé sur les principes value
- Analyse de consistance des performances
- Export des résultats

### 5. Gestionnaire de Portefeuille
- Suivi des positions
- Analyse de performance
- Optimisation d'allocation
- Recommandations personnalisées

## 📋 Prérequis

- Python 3.8+
- pip

## 🛠️ Installation

1. Cloner le repository
```bash
git clone https://github.com/yourusername/financial-dashboard.git
cd financial-dashboard
```

2. Installer les dépendances
```bash
pip install -r requirements.txt
```

3. Lancer l'application
```bash
streamlit run main.py
```

## 📁 Structure du Projet

```
financial-dashboard/
│
├── main.py                 # Point d'entrée de l'application
├── requirements.txt        # Dépendances Python
├── README.md              # Documentation
│
├── core/                  # Logique métier
│   ├── __init__.py
│   ├── market_data.py     # Récupération des données
│   ├── analysis.py        # Calculs financiers
│   ├── dividend_analysis.py # Analyse des dividendes
│   ├── news.py           # Gestion des actualités
│   └── portfolio.py      # Gestion de portefeuille
│
└── ui_components/        # Interface utilisateur
    ├── __init__.py
    ├── sidebar.py        # Barre latérale
    ├── technical_analysis.py # Onglet analyse technique
    ├── fundamental_analysis.py # Onglet analyse fondamentale
    ├── news.py          # Onglet actualités
    ├── screener.py      # Onglet screener
    ├── portfolio.py     # Onglet portefeuille
    ├── styles.py        # Styles CSS
    ├── metrics.py       # Composants métriques
    └── common.py        # Fonctions communes
```

## 🔧 Configuration

L'application utilise le fichier CSV hébergé sur GitHub pour la structure du marché :
`https://raw.githubusercontent.com/Culass31/actions/refs/heads/main/actions.csv`

## 💡 Utilisation

### Sélection d'actions
1. Choisir une méthode de filtrage (géographique, sectorielle, marché)
2. Sélectionner les critères souhaités
3. Choisir une action dans la liste filtrée

### Analyse technique
- Sélectionner la période d'analyse
- Activer/désactiver les indicateurs techniques
- Consulter les recommandations automatiques

### Analyse fondamentale
- Consulter les métriques financières
- Visualiser la valorisation intrinsèque
- Comparer avec les objectifs des analystes

### Screener
1. Définir les critères de screening
2. Lancer l'analyse
3. Consulter les résultats triés par score
4. Exporter les données

### Portefeuille
1. Ajouter des positions manuellement ou via CSV
2. Analyser la performance globale
3. Obtenir des recommandations d'optimisation

## 🔍 Algorithmes et Méthodes

### Valorisation intrinsèque
- **DCF** : Actualisation des flux de trésorerie futurs
- **Graham** : Formule de Benjamin Graham
- **PER sectoriel** : Comparaison avec le secteur
- **Valeur patrimoniale** : Actifs nets tangibles

### Score Buffett-Graham
- ROE et stabilité des bénéfices
- Marges et croissance constantes
- Endettement limité
- Valorisation raisonnable

### Optimisation de portefeuille
- Pondération basée sur les recommandations
- Ajustement selon le profil de risque
- Prise en compte du potentiel de hausse

## 📊 Sources de Données

- **Cours et données financières** : Yahoo Finance (yfinance)
- **Structure du marché** : CSV personnalisé
- **Actualités** : Yahoo Finance News API

## ⚠️ Avertissement

Ce dashboard est fourni à titre informatif uniquement et ne constitue pas un conseil en investissement. Les utilisateurs doivent effectuer leurs propres recherches avant de prendre des décisions d'investissement.

## 📄 Licence

MIT License

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche feature
3. Commit vos changements
4. Push sur la branche
5. Ouvrir une Pull Request

## 📧 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.
