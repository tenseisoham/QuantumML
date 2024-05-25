# %%
TOKEN="use token here"
# %%
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import QuantumCircuit
from qiskit import transpile

service = QiskitRuntimeService(channel="ibm_quantum", instance='ibm-q/open/main', token=TOKEN)

# %%
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# %%
backend = service.least_busy(simulator=False, operational=True)

# %%
# Convert to an ISA circuit and layout-mapped observables.
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
# isa_circuit = pm.run(qc)
# print("Optimized circuit depth:", isa_circuit.depth())
# isa_circuit.draw('mpl')

# %%
# estimator = Estimator(backend=backend)
# estimator.options.resilience_level = 1
# estimator.options.default_shots = 5000
 
# mapped_observables = [
#     observable.apply_layout(isa_circuit.layout) for observable in observables
# ]
 
# # One pub, with one circuit to run against five different observables.
# job = estimator.run([(isa_circuit, mapped_observables)])
 
# Use the job ID to retrieve your job data later
# print(f">>> Job ID: {job.job_id()}")

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

breastcancer = '/Users/sanidhya/Downloads/diabetes.csv'
df = pd.read_csv(breastcancer)

# Reduce the dataframe size by sampling 1/3 of the data
df = df.sample(frac=1/3, random_state=1)  # random_state for reproducibility

# Separate the dataset into features (X) and target label (y)
y = df['Outcome']  # Target label: diagnosis
X = df.drop('Outcome', axis=1)  # Features: all other columns

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Set parameters for the quantum feature map
feature_dimension = 2  # Number of features used in the quantum feature map
reps = 2  # Number of repetitions of the feature map circuit
entanglement = 'linear'  # Type of entanglement in the quantum circuit

# %%
from sklearn.preprocessing import MinMaxScaler

X = MinMaxScaler().fit_transform(X)

# %%
from sklearn.model_selection import train_test_split
from qiskit_algorithms.utils import algorithm_globals

algorithm_globals.random_seed = 123
train_features, test_features, train_labels, test_labels = train_test_split(
    X, y, train_size=0.8, random_state=algorithm_globals.random_seed
)

# %%
from sklearn.svm import SVC

svc = SVC()
_ = svc.fit(train_features, train_labels)  # suppress printing the return value

# %%
train_score_c4 = svc.score(train_features, train_labels)
test_score_c4 = svc.score(test_features, test_labels)

print(f"Classical SVC on the training dataset: {train_score_c4:.2f}")
print(f"Classical SVC on the test dataset:     {test_score_c4:.2f}")

# %%
from qiskit.circuit.library import ZZFeatureMap

num_features = X.shape[1]

print("Num of features in datatset:", num_features)

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
feature_map.decompose().draw(output="mpl", style="clifford", fold=20)

#%%

# Transpile your feature map for the specific backend
transpiled_feature_map = transpile(feature_map, backend=backend, optimization_level=3)

# Now you can visualize or use the transpiled_feature_map
# transpiled_feature_map.draw(output='mpl')

# %%
from qiskit.circuit.library import RealAmplitudes

# Assuming num_features is the correct number of qubits you determined
# num_features = transpiled_feature_map.num_qubits  # Update to match feature map post-transpilation
ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
# # ansatz.decompose().draw(output="mpl", style="clifford", fold=20)
transplied_ansatz = transpile(ansatz, backend=backend, optimization_level=3)

# %%
from qiskit_algorithms.optimizers import COBYLA

optimizer = COBYLA(maxiter=100)

# %%
# from qiskit.primitives import Sampler

# sampler = Sampler()

# %%
sampler = Sampler(backend=backend)

# %%
from matplotlib import pyplot as plt
from IPython.display import clear_output

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

# %%
import time
from qiskit_machine_learning.algorithms.classifiers import VQC

vqc = VQC(
    sampler=sampler,
    # feature_map=feature_map,
    feature_map=transpiled_feature_map,
    # ansatz=ansatz,
    ansatz=transplied_ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)

# clear objective value history
objective_func_vals = []

start = time.time()
vqc.fit(train_features, train_labels)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")

# %%
train_score_q4 = vqc.score(train_features, train_labels)
test_score_q4 = vqc.score(test_features, test_labels)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")

# %%
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns

features = PCA(n_components=2).fit_transform(X)

plt.rcParams["figure.figsize"] = (6, 6)
sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=y, palette="tab10")

# %%
from sklearn.model_selection import train_test_split
from qiskit_algorithms.utils import algorithm_globals
from sklearn.svm import SVC

train_features, test_features, train_labels, test_labels = train_test_split(
    features, y, train_size=0.8, random_state=algorithm_globals.random_seed
)

svc = SVC()

svc.fit(train_features, train_labels)

train_score_c2 = svc.score(train_features, train_labels)
test_score_c2 = svc.score(test_features, test_labels)

print(f"Classical SVC on the training dataset: {train_score_c2:.2f}")
print(f"Classical SVC on the test dataset:     {test_score_c2:.2f}")

# %%
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplites

num_features = features.shape[1]

print(num_features)

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
ansatz = RealAmplitudes(num_qubits=num_features, reps=3)

# %%
from qiskit_algorithms.optimizers import COBYLA

optimizer = COBYLA(maxiter=40)

# %%
from qiskit.primitives import Sampler

sampler = Sampler()

# %%
from matplotlib import pyplot as plt
from IPython.display import clear_output

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

# %%
import time
from qiskit_machine_learning.algorithms.classifiers import VQC

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)

# clear objective value history
objective_func_vals = []

# make the objective function plot look nicer.
plt.rcParams["figure.figsize"] = (12, 6)

start = time.time()
vqc.fit(train_features, train_labels)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")

# %%
train_score_q2_ra = vqc.score(train_features, train_labels)
test_score_q2_ra = vqc.score(test_features, test_labels)

print(f"Quantum VQC on the training dataset using RealAmplitudes: {train_score_q2_ra:.2f}")
print(f"Quantum VQC on the test dataset using RealAmplitudes:     {test_score_q2_ra:.2f}")

# %%
from qiskit.circuit.library import EfficientSU2

ansatz = EfficientSU2(num_qubits=num_features, reps=3)
optimizer = COBYLA(maxiter=40)

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)

# clear objective value history
objective_func_vals = []

start = time.time()
vqc.fit(train_features, train_labels)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")

# %%
train_score_q2_eff = vqc.score(train_features, train_labels)
test_score_q2_eff = vqc.score(test_features, test_labels)

print(f"Quantum VQC on the training dataset using EfficientSU2: {train_score_q2_eff:.2f}")
print(f"Quantum VQC on the test dataset using EfficientSU2:     {test_score_q2_eff:.2f}")

# %%
print(f"Model                           | Train Score | Test Score")
print(f"SVC, 8 features                 | {train_score_c4:10.2f} | {test_score_c4:10.2f}")
print(f"VQC, 4 features, RealAmplitudes | {train_score_q4:10.2f} | {test_score_q4:10.2f}")
print(f"----------------------------------------------------------")
print(f"SVC, 2 features                 | {train_score_c2:10.2f} | {test_score_c2:10.2f}")
print(f"VQC, 2 features, RealAmplitudes | {train_score_q2_ra:10.2f} | {test_score_q2_ra:10.2f}")
print(f"VQC, 2 features, EfficientSU2   | {train_score_q2_eff:10.2f} | {test_score_q2_eff:10.2f}")