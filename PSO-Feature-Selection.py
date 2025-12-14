import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X, y = make_classification(n_samples=1000, n_features=5, n_informative=3,
                           n_redundant=0, n_repeated=0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

w = 0.7
c1 = 2
c2 = 2


n_particles = 10
n_iterations = 20
n_features = 5

particles_position = np.random.randint(2, size=(n_particles, n_features))
particles_velocity = np.random.uniform(low=-1, high=1, size=(n_particles, n_features))

pBest_position = particles_position.copy()
pBest_fitness = np.zeros(n_particles)
gBest_position = np.zeros(n_features)
gBest_fitness = 0

def calculate_fitness(mask, X_train, X_test, y_train, y_test):
    if np.count_nonzero(mask) == 0:
        return 0

    selected_features_train = X_train[:, mask == 1]
    selected_features_test = X_test[:, mask == 1]

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(selected_features_train, y_train)
    predictions = clf.predict(selected_features_test)
    return accuracy_score(y_test, predictions)

print(f"Starting PSO Optimization on {n_features} features...\n")

for t in range(n_iterations):

    for i in range(n_particles):
        current_fitness = calculate_fitness(particles_position[i], X_train, X_test, y_train, y_test)

        if current_fitness > pBest_fitness[i]:
            pBest_fitness[i] = current_fitness
            pBest_position[i] = particles_position[i].copy()

        if current_fitness > gBest_fitness:
            gBest_fitness = current_fitness
            gBest_position = particles_position[i].copy()

    for i in range(n_particles):

        r1 = np.random.rand(n_features)
        r2 = np.random.rand(n_features)

        new_velocity = (w * particles_velocity[i]) + \
                       (c1 * r1 * (pBest_position[i] - particles_position[i])) + \
                       (c2 * r2 * (gBest_position - particles_position[i]))

        particles_velocity[i] = new_velocity

        sigmoid_v = 1 / (1 + np.exp(-new_velocity))

        random_thresholds = np.random.rand(n_features)
        particles_position[i] = (sigmoid_v > random_thresholds).astype(int)

    print(f"Iteration {t+1}/{n_iterations} - Best Accuracy: {gBest_fitness:.4f} - Features: {gBest_position}")

print("\n" + "="*40)
print("OPTIMIZATION COMPLETE")
print("="*40)
print(f"Best Accuracy Achieved: {gBest_fitness*100:.2f}%")
print(f"Selected Feature Vector: {gBest_position}")

selected_indices = [i for i, x in enumerate(gBest_position) if x == 1]
print(f"Indices of Selected Features: {selected_indices}")

