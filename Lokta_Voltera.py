import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paramètres du modèle
alpha = 2 / 3
beta = 4 / 3
delta = 1
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
renard = np.array(renard) * 2000

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

# Étape 3 : Comparaison modèle vs données réelles
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

#Etape 4 : MSE
