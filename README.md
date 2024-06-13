# Quantum Feature Maps for Enhanced Kernel Methods in Machine Learning

This repository contains the implementation and examples of kernel-based methods applied to quantum machine learning. It includes various techniques for leveraging quantum computing to enhance machine learning algorithms, particularly focusing on the application of quantum-enhanced kernels in Support Vector Machines (SVMs).

## Description

Kernel-based methods are a cornerstone of classical machine learning, and quantum computing offers a new perspective and computational advantage in dealing with these methods. In this project, we explore the theoretical underpinnings and practical applications of utilizing quantum Hilbert spaces as feature spaces for kernel-based ML models.

## Features

- Implementation of quantum feature maps
- Construction of quantum kernels
- Integration with classical SVM frameworks
- Examples using simulated quantum computing environments
- Performance analysis and comparisons with classical approaches

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/quantum-feature-maps.git
cd quantum-feature-maps
pip install -r requirements.txt
```

## Usage

### Quantum Feature Maps

The core of this implementation involves a hybrid quantum-classical approach where classical data is encoded into a quantum state, and quantum mechanics is employed to compute a similarity matrix (kernel) used in a classical support vector machine algorithm.

### Example Code

Here is an example of how to construct a quantum feature map and use it in an SVM:

```python

```python
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define quantum feature map and quantum kernel
feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2)
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=Aer.get_backend('statevector_simulator'))

# Train QSVC model
qsvc = QSVC(quantum_kernel=quantum_kernel)
qsvc.fit(X_train, y_train)

# Evaluate the model
score = qsvc.score(X_test, y_test)
print(f"Test accuracy: {score:.2f}")
```

## Performance Analysis

### Results

The performance of the quantum-enhanced SVM is compared with classical SVM on several datasets. Below are some of the results:

**Breast Cancer Dataset:**

![Breast Cancer Results](images/breast_cancer_results.png)

**Diabetes Dataset:**

![Diabetes Results](images/diabetes_results.png)


## Conclusion

Our experiments demonstrated that quantum kernels could achieve comparable, if not superior, classification accuracy to classical kernels, particularly for non-linearly separable data. The combination of the Z-Feature map, EfficientSU2 ansatz, and L_BFGS_B optimizer showed the most promise, often outperforming classical SVM in various metrics.

## Future Work

Further exploration of quantum feature maps and their application to other machine learning models is a promising area for future research. Optimizing quantum circuits and incorporating error mitigation techniques will be crucial as quantum hardware continues to evolve.

## References

- [Qiskit Machine Learning](https://qiskit.org/documentation/machine-learning/)
- [Supervised Learning with Quantum Enhanced Feature Spaces](https://arxiv.org/abs/1804.11326)



Plaksha University, May 30, 2024
