"""
=============================================================================
ANALYSE COMPLÈTE DE LA BASE DE DONNÉES WINE QUALITY
=============================================================================
Dataset: UCI ML Repository - Wine Quality (ID: 186)
Objectif: Analyse descriptive complète avec visualisations
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

print("\n" + "="*80)
print("CHARGEMENT DU DATASET WINE QUALITY")
print("="*80)

# Fetch dataset
wine_quality = fetch_ucirepo(id=186)

# Extract data
X = wine_quality.data.features
y = wine_quality.data.targets

print("\n✓ Dataset chargé avec succès")
print(f"  Features (X): {X.shape}")
print(f"  Target (y):   {y.shape}")

# =============================================================================
# MÉTADONNÉES
# =============================================================================

print("\n" + "="*80)
print("MÉTADONNÉES DU DATASET")
print("="*80)

metadata = wine_quality.metadata
print(f"\nNom du dataset: {metadata['name']}")
print(f"Description: {metadata.get('abstract', 'N/A')}")

# =============================================================================
# INFORMATIONS SUR LES VARIABLES
# =============================================================================

print("\n" + "="*80)
print("VARIABLES DU DATASET")
print("="*80)

variables_info = wine_quality.variables
print("\n", variables_info)

# =============================================================================
# EXPLORATION INITIALE
# =============================================================================

print("\n" + "="*80)
print("EXPLORATION INITIALE DES DONNÉES")
print("="*80)

print("\nPremières lignes des features (X):")
print(X.head(10))

print("\nPremières lignes de la cible (y):")
print(y.head(10))

print("\nInformations sur X:")
X.info()

print("\nInformations sur y:")
y.info()

# =============================================================================
# STATISTIQUES DESCRIPTIVES - FEATURES
# =============================================================================

print("\n" + "="*80)
print("STATISTIQUES DESCRIPTIVES DES FEATURES")
print("="*80)

stats_X = X.describe()
print("\n", stats_X.T)

# Statistiques supplémentaires
print("\n" + "-"*80)
print("STATISTIQUES COMPLÉMENTAIRES")
print("-"*80)

for col in X.columns:
    print(f"\n{col}:")
    print(f"  Min        : {X[col].min():.4f}")
    print(f"  Max        : {X[col].max():.4f}")
    print(f"  Range      : {X[col].max() - X[col].min():.4f}")
    print(f"  Mean       : {X[col].mean():.4f}")
    print(f"  Median     : {X[col].median():.4f}")
    print(f"  Std Dev    : {X[col].std():.4f}")
    print(f"  Variance   : {X[col].var():.
