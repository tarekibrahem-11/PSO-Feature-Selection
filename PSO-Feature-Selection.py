import numpy as np
import random
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_data, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y, test_size=0.3, random_state=42
)


w = 0.7
c1 = 2
c2 = 2
num_particles = 10
num_iterations = 20
num_features = 5


def sigmoid(v):
    return 1 / (1 + np.exp(-np.clip(v, -20, 20)))

def calculate_fitness(mask):
    if np.count_nonzero(mask) == 0:
        return 0

    X_train_sel = X_train[:, mask == 1]
    X_test_sel = X_test[:, mask == 1]

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train_sel, y_train)
    preds = clf.predict(X_test_sel)

    return accuracy_score(y_test, preds)


X = np.random.randint(2, size=(num_particles, num_features))
V = np.random.uniform(-1, 1, size=(num_particles, num_features))

fitness = np.array([calculate_fitness(X[i]) for i in range(num_particles)])

pbest = X.copy()
pbest_fitness = fitness.copy()

gbest_idx = np.argmax(pbest_fitness)
gbest = pbest[gbest_idx].copy()
gbest_fitness = pbest_fitness[gbest_idx]

print(f"Initial Global Best: {gbest} Accuracy: {gbest_fitness:.4f}\n")


for iteration in range(1, num_iterations + 1):

    for i in range(num_particles):
        r1 = random.random()
        r2 = random.random()

        V[i] = (
            w * V[i]
            + c1 * r1 * (pbest[i] - X[i])
            + c2 * r2 * (gbest - X[i])
        )

        prob = sigmoid(V[i])
        for d in range(num_features):
            X[i][d] = 1 if random.random() < prob[d] else 0

    current_fitness = np.array([calculate_fitness(X[i]) for i in range(num_particles)])

    improved = current_fitness > pbest_fitness
    pbest_fitness[improved] = current_fitness[improved]
    pbest[improved] = X[improved]

    best_idx = np.argmax(current_fitness)
    if current_fitness[best_idx] > gbest_fitness:
        gbest_fitness = current_fitness[best_idx]
        gbest = X[best_idx].copy()

    print(f"Iteration {iteration}/{num_iterations}")
    print(f"Best Accuracy: {gbest_fitness:.4f}")
    print(f"Best Feature Mask: {gbest}\n")


selected_features = [i for i, v in enumerate(gbest) if v == 1]

print("="*40)
print("OPTIMIZATION COMPLETE")
print("="*40)
print(f"Best Accuracy: {gbest_fitness*100:.2f}%")
print(f"Selected Feature Vector: {gbest}")
print(f"Selected Feature Indices: {selected_features}")
