import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

# Paramètres du modèle
alpha = 1
beta = 2 / 3
delta = 2 / 3
gamma = 1
step = 0.001

# Initialisation des données
time = [0]
lapin = [1]
renard = [2]

# Simulation du modèle
for _ in range(0, 100_000):
    new_value_time = time[-1] + step
    new_value_lapin = (lapin[-1] * (alpha - beta * renard[-1])) * step + lapin[-1]
    new_value_renard = (renard[-1] * (delta * lapin[-1] - gamma)) * step + renard[-1]

    time.append(new_value_time)
    lapin.append(new_value_lapin)
    renard.append(new_value_renard)

# Mise à l'échelle des populations
lapin = np.array(lapin) * 1000
renard = np.array(renard) * 1000

# Visualisation des populations simulées
plt.figure(figsize=(13, 10))
plt.plot(time, lapin, "b-", label="Lapins")
plt.plot(time, renard, "r-", label="Renards")
plt.xlabel("Temps (Mois)")
plt.ylabel("Population")
plt.title("Dynamique des populations Proie-Prédateur")
plt.legend()
plt.show()

# Lecture et analyse des données réelles
df = pd.read_csv("populations_lapins_renards.csv", delimiter=",")
# Contenu du fichier CSV
print(df.head())
# Résumé des données
print(df.info())

# Conversion des dates
print("\nConversion des dates en format datetime...")
df["date"] = pd.to_datetime(df["date"], yearfirst=True, errors="coerce")
print(df.info())

# Visualisation des données réelles
plt.figure(figsize=(13, 10))
plt.plot(df["date"], df["lapin"], "b--", label="Lapins (réels)")
plt.plot(df["date"], df["renard"], "r--", label="Renards (réels)")
plt.xlabel("Date")
plt.ylabel("Population")
plt.title("Données réelles des populations Proie-Prédateur")
plt.legend()
plt.show()

#Comparaison modèle vs données réelles
plt.figure(figsize=(13, 10))
plt.plot(time[: len(df)], lapin[: len(df)], "b-", label="Lapins (modèle)")
plt.plot(time[: len(df)], renard[: len(df)], "r-", label="Renards (modèle)")
plt.plot(df["date"], df["lapin"], "b--", label="Lapins (réels)")
plt.plot(df["date"], df["renard"], "r--", label="Renards (réels)")
plt.xlabel("Temps")
plt.ylabel("Population")
plt.title("Comparaison : Modèle Lotka-Volterra vs Données Réelles")
plt.legend()
plt.show()

#MSE

# Fonction Lotka-Volterra
def lotka_volterra_step(lapin, renard, alpha, beta, delta, gamma, step):
    """
    Effectue un pas de simulation pour le modèle Lotka-Volterra.
    
    Args:
        lapin (float): Population des lapins à l'instant courant.
        renard (float): Population des renards à l'instant courant.
        alpha (float): Taux de croissance des lapins.
        beta (float): Taux de prédation des lapins par les renards.
        delta (float): Taux de reproduction des renards en fonction des lapins consommés.
        gamma (float): Taux de mortalité des renards.
        steps (float): Taille du pas de simulation.
    
    Returns:
        tuple: Nouvelles populations des lapins et des renards.
    """
    new_lapin = (lapin * (alpha - beta * renard)) * step + lapin
    new_renard = (renard * (delta * lapin - gamma)) * step + renard
    return new_lapin, new_renard

# Fonction pour générer des prédictions avec le modèle Lotka-Volterra
def predict_lotka_volterra(time_steps, lapin_init, renard_init, alpha, beta, delta, gamma, step):
    """
    Génère les prédictions des populations des lapins et des renards.

    Args:
        time_steps (int): Nombre total de pas de temps.
        lapin_init (float): Population initiale des lapins.
        renard_init (float): Population initiale des renards.
        alpha, beta, delta, gamma (float): Paramètres du modèle Lotka-Volterra.
        steps (float): Taille du pas de temps.

    Returns:
        tuple: (time, lapins, renards) où chaque élément est une liste.
    """
    time = [0]
    lapins = [lapin_init]
    renards = [renard_init]
    
    for _ in range(time_steps):
        new_time = time[-1] + step
        new_lapin, new_renard = lotka_volterra_step(lapins[-1], renards[-1], alpha, beta, delta, gamma, step)
        time.append(new_time)
        lapins.append(new_lapin)
        renards.append(new_renard)
    
    return np.array(time), np.array(lapins), np.array(renards)

# Fonction objectif : calcul de l'erreur quadratique moyenne (MSE)
def mse_objective(real_lapins, real_renards, predicted_lapins, predicted_renards):
    """
    Calcule l'erreur quadratique moyenne (MSE) entre les données réelles et les prédictions.

    Args:
        real_lapins (array-like): Données réelles pour les lapins.
        real_renards (array-like): Données réelles pour les renards.
        predicted_lapins (array-like): Prédictions du modèle pour les lapins.
        predicted_renards (array-like): Prédictions du modèle pour les renards.

    Returns:
        float: Erreur quadratique moyenne (MSE).
    """
    mse_lapins = np.mean((real_lapins - predicted_lapins) ** 2)
    mse_renards = np.mean((real_renards - predicted_renards) ** 2)
    return mse_lapins + mse_renards

# Exemple d'utilisation
# Données réelles (exemple fictif)
time_real = np.linspace(0, 20, 100)
real_lapins = np.sin(time_real) + 0.1
real_renards = np.cos(time_real) + 0.1

# Générer des prédictions (best_params trouvé après le grid_search )
alpha, beta, delta, gamma = 1/3, 4/3, 4/3, 1/3
time_steps = len(time_real) - 1
step = (time_real[-1] - time_real[0]) / time_steps

_, predicted_lapins, predicted_renards = predict_lotka_volterra(
    time_steps, real_lapins[1], real_renards[2], alpha, beta, delta, gamma, step
)

# Calcul de la MSE
error = mse_objective(real_lapins, real_renards, predicted_lapins, predicted_renards)
print(f"Erreur quadratique moyenne (MSE) après GridSearch: {error:.4f}")

#Utilisation du GridSearch

# Paramètres possibles pour le Grid Search
alpha_values = [1/3, 2/3, 1, 4/3]
beta_values = [1/3, 2/3, 1, 4/3]
delta_values = [1/3, 2/3, 1, 4/3]
gamma_values = [1/3, 2/3, 1, 4/3]

# Liste pour stocker les résultats
results = []
step = 0.001
time_steps = len(time_real) - 1

# Grid Search
for alpha in alpha_values:
    for beta in beta_values:
        for delta in delta_values:
            for gamma in gamma_values:
                # Générer des prédictions
                _, predicted_lapins, predicted_renards = predict_lotka_volterra(
                    time_steps, real_lapins[1], real_renards[2], alpha, beta, delta, gamma, step
                )
                
                # Calculer la MSE
                error = mse_objective(real_lapins, real_renards, predicted_lapins, predicted_renards)
                
                # Ajouter les résultats
                results.append({
                    "alpha": alpha,
                    "beta": beta,
                    "delta": delta,
                    "gamma": gamma,
                    "mse": error
                })

# Trier les résultats par MSE croissant
results = sorted(results, key=lambda x: x["mse"])

# Exporter les résultats dans un fichier CSV
results = sorted(results, key=lambda x: x["mse"])
output_file = "grid_search_results.csv"
with open(output_file, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["alpha", "beta", "delta", "gamma", "mse"])
    writer.writeheader()
    writer.writerows(results)

print(f"Résultats du Grid Search exportés dans : {output_file}")

# Afficher les meilleurs paramètres
best_result = results[0]
print("Meilleurs paramètres trouvés :")
print(f"Alpha : {best_result['alpha']}, Beta : {best_result['beta']}, Delta : {best_result['delta']}, Gamma : {best_result['gamma']}")
print(f"MSE minimale : {best_result['mse']:.4f}")




























