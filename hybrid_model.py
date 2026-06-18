import torch
import torch.nn as nn
import pennylane as qml

# ==============================
# Quantum Device
# ==============================
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# ==============================
# Quantum Circuit
# ==============================
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):

    # Ensure correct dtype
    inputs = inputs.to(torch.float32)
    weights = weights.to(torch.float32)

    # Angle Encoding
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    # Variational Layer
    for i in range(n_qubits):
        qml.RZ(weights[i], wires=i)

    # Entanglement
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# ==============================
# Hybrid Model
# ==============================
class HybridModel(nn.Module):

    def __init__(self):
        super(HybridModel, self).__init__()

        # --------------------------
        # CNN Feature Extractor
        # --------------------------
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 14 * 14, n_qubits)
        )

        # --------------------------
        # Quantum Parameters
        # --------------------------
        self.q_weights = nn.Parameter(torch.randn(n_qubits))

        # --------------------------
        # Classical Output Layer
        # --------------------------
        self.fc = nn.Linear(n_qubits, 1)

    # ==============================
    # Forward Pass
    # ==============================
    def forward(self, x):

        # CNN Feature Extraction
        features = self.cnn(x)

        # Quantum Layer
        q_out = []

        for f in features:
            q_result = quantum_circuit(f, self.q_weights)

            # Convert to FloatTensor safely
            q_result = torch.stack(q_result).to(torch.float32)

            q_out.append(q_result)

        q_out = torch.stack(q_out).to(x.device)

        # Classical Classification
        output = torch.sigmoid(self.fc(q_out))

        return output