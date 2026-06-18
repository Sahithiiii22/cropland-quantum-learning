import torch
import torch.nn as nn
import pennylane as qml

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):

    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    for i in range(n_qubits):
        qml.RZ(weights[i], wires=i)

    for i in range(n_qubits-1):
        qml.CNOT(wires=[i, i+1])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class HybridModel(nn.Module):

    def __init__(self):
        super().__init__()

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, n_qubits)
        )

        self.q_weights = nn.Parameter(torch.randn(n_qubits))
        self.fc = nn.Linear(n_qubits, 1)

    def forward(self, x):

        features = self.cnn(x)

        q_out = torch.stack([
            torch.tensor(quantum_circuit(f, self.q_weights))
            for f in features
        ])

        output = torch.sigmoid(self.fc(q_out))

        return output