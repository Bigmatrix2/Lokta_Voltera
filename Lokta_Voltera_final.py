import matplotlib.pyplot as plt
import numpy
import pandas
import math
# import itertools
# from tqdm import tqdm
from tqdm.contrib.itertools import product


def lokta_voltera_forecast(alpha, beta, delta, gama, step, nb_jours):
	time = [0]
	lapin = [1]
	renard = [2]
	
	nb_points = int(nb_jours / step)
	for _ in range(1, nb_points):
		new_value_time = time[-1] + step
		new_value_lapin = (lapin[-1] * (alpha - beta * renard[-1])) * step + lapin[-1]
		new_value_renard = (renard[-1] * (delta * lapin[-1] - gama)) * step + renard[-1]

		time.append(new_value_time)
		lapin.append(new_value_lapin)
		renard.append(new_value_renard)

	time = time[::1000]
	lapin = numpy.array(lapin[::1000]) * 1000
	renard = numpy.array(renard[::1000]) * 1000
	return time, lapin, renard


def plot_data(time, lapin, renard, ground_truth):
	plt.figure(figsize=(15, 6))
	plt.plot(time, lapin, "b:", label='Lapins prediction')
	plt.plot(time, renard, "r:", label='Renards prediction')
	plt.plot(time, ground_truth["lapin"], "bo", label='Lapins réel')
	plt.plot(time, ground_truth["renard"], "ro", label='Renards réel')

	plt.xlabel('Temps (Mois)')
	plt.ylabel('Population')
	plt.title('Dynamique des populations Proie-Prédateur')
	plt.legend()
	plt.show()


def rmse(model_prediction, valeur_reel):

	nb_echantillions = len(model_prediction)

	total_squared_errors = 0
	for index in range(nb_echantillions):
		total_squared_errors += (model_prediction[index] - valeur_reel[index]) ** 2

	return math.sqrt(total_squared_errors / nb_echantillions)


def grid_search(step, nb_jours, ground_truth):
	alphas = numpy.linspace(0.1, 1, 4)
	betas = numpy.linspace(0.1, 1, 4)
	deltas = numpy.linspace(0.1, 1, 4)
	gamas = numpy.linspace(0.1, 1, 4)

	best_alpha, best_beta, best_delta, best_gama = None, None, None, None 
	best_rmse = float("inf")

	for alpha, beta, delta, gama in product(alphas, betas, deltas, gamas):
		time, lapin, renard = lokta_voltera_forecast(alpha, beta, delta, gama, step, nb_jours)
		rmse_lapin = rmse(lapin, ground_truth["lapin"].values)
		rmse_renard = rmse(renard, ground_truth["renard"].values)
		actual_rmse = rmse_lapin + rmse_renard
		if actual_rmse < best_rmse:
			best_rmse = actual_rmse
			best_alpha, best_beta, best_delta, best_gama = alpha, beta, delta, gama

	print(best_alpha, best_beta, best_delta, best_gama)
	time, best_lapin, best_renard = lokta_voltera_forecast(best_alpha, best_beta, best_delta, best_gama, step, nb_jours)
	plot_data(time, best_lapin, best_renard, ground_truth)

	return time, best_lapin, best_renard

if __name__ == "__main__":

	step = 0.001 # nombre de points par jours
	nb_jours = 1000
	ground_truth = pandas.read_csv("populations_lapins_renards.csv")

	grid_search(step, nb_jours, ground_truth)