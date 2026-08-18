import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

SEQ_LEN = 5
NUM_FEATURES = 15

MODEL_PATH = "/opt/spark-models/Tham_lstm_attn_checkpoint.pth"

THRESHOLD = 0.5

DEVICE = torch.device("cpu")


# ============================================================
# ATTENTION
# ============================================================

class Attention(torch.nn.Module):
    """
    Attention mechanism.

    Input:
        output  : (batch, output_len, dim)
        context : (batch, input_len, dim)

    Output:
        output  : (batch, output_len, dim)
        attn    : (batch, output_len, input_len)
    """

    def __init__(self, dim):

        super(Attention, self).__init__()

        self.linear_out = torch.nn.Linear(
            dim * 2,
            dim
        )

        self.mask = None


    def set_mask(self, mask):

        self.mask = mask


    def forward(self, output, context):

        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)

        # ----------------------------------------------------
        # Attention score
        #
        # output:
        #   (batch, output_len, dim)
        #
        # context:
        #   (batch, input_len, dim)
        #
        # attn:
        #   (batch, output_len, input_len)
        # ----------------------------------------------------

        attn = torch.bmm(
            output,
            context.transpose(1, 2)
        )

        if self.mask is not None:

            attn.data.masked_fill_(
                self.mask,
                -float("inf")
            )

        attn = F.softmax(
            attn.view(-1, input_size),
            dim=1
        ).view(
            batch_size,
            -1,
            input_size
        )

        # ----------------------------------------------------
        # Weighted context
        # ----------------------------------------------------

        mix = torch.bmm(
            attn,
            context
        )

        # ----------------------------------------------------
        # Concatenate context + output
        # ----------------------------------------------------

        combined = torch.cat(
            (
                mix,
                output
            ),
            dim=2
        )

        # ----------------------------------------------------
        # Linear projection
        # ----------------------------------------------------

        output = F.tanh(
            self.linear_out(
                combined.view(
                    -1,
                    2 * hidden_size
                )
            )
        ).view(
            batch_size,
            -1,
            hidden_size
        )

        return output, attn


# ============================================================
# FRAUD LSTM WITH ATTENTION
# ============================================================

class FraudLSTMWithAttention(torch.nn.Module):

    def __init__(
        self,
        num_features,
        hidden_size=100,
        hidden_size_lstm=100,
        num_layers_lstm=1,
        dropout_lstm=0,
        attention_out_dim=100
    ):

        super(FraudLSTMWithAttention, self).__init__()

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        self.num_features = num_features

        self.hidden_size = hidden_size

        # ----------------------------------------------------
        # LSTM
        #
        # Input:
        #   (batch, seq_len, num_features)
        #
        # Output:
        #   (batch, seq_len, hidden_size_lstm)
        # ----------------------------------------------------

        self.lstm = torch.nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size_lstm,
            num_layers=num_layers_lstm,
            batch_first=True,
            dropout=dropout_lstm
        )

        # ----------------------------------------------------
        # Project last transaction
        # ----------------------------------------------------

        self.ff = torch.nn.Linear(
            num_features,
            hidden_size_lstm
        )

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.attention = Attention(
            attention_out_dim
        )

        # ----------------------------------------------------
        # Fully connected
        # ----------------------------------------------------

        self.fc1 = torch.nn.Linear(
            hidden_size_lstm,
            hidden_size
        )

        self.relu = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(
            hidden_size,
            1
        )

        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):

        # ====================================================
        # INPUT
        #
        # x:
        #   (batch, seq_len, num_features)
        #
        # Example:
        #   (1, 5, 15)
        # ====================================================

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

        hidden_states, _ = self.lstm(x)

        # hidden_states:
        #
        # (batch, 5, 100)

        # ----------------------------------------------------
        # Last transaction
        # ----------------------------------------------------

        last_transaction = x[:, -1, :]

        # (batch, 15)

        # ----------------------------------------------------
        # Project last transaction
        # ----------------------------------------------------

        context_vector = self.ff(
            last_transaction
        )

        # (batch, 100)

        # ----------------------------------------------------
        # Add output sequence dimension
        # ----------------------------------------------------

        context_vector = context_vector.unsqueeze(1)

        # (batch, 1, 100)

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        combined_state, attn = self.attention(
            context_vector,
            hidden_states
        )

        # combined_state:
        #
        # (batch, 1, 100)

        # ----------------------------------------------------
        # Remove sequence dimension
        # ----------------------------------------------------

        combined_state = combined_state[:, 0, :]

        # (batch, 100)

        # ----------------------------------------------------
        # Fully connected
        # ----------------------------------------------------

        hidden = self.fc1(
            combined_state
        )

        hidden = self.relu(
            hidden
        )

        output = self.fc2(
            hidden
        )

        output = self.sigmoid(
            output
        )

        return output


# ============================================================
# CREATE MODEL
# ============================================================

print("=" * 80)
print("LSTM ATTENTION INFERENCE TEST")
print("=" * 80)

print()
print("Creating model...")

model = FraudLSTMWithAttention(
    num_features=NUM_FEATURES,
    hidden_size=100,
    hidden_size_lstm=100,
    num_layers_lstm=1,
    dropout_lstm=0,
    attention_out_dim=100
)

print("Model created successfully.")


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("Loading checkpoint...")
print(f"Model path: {MODEL_PATH}")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

print("Checkpoint loaded successfully.")

print()
print("Checkpoint keys:")

for key in checkpoint.keys():

    print(f"  - {key}")


# ============================================================
# LOAD MODEL WEIGHTS
# ============================================================

print()
print("Loading model_state_dict...")

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("Model weights loaded successfully.")


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model.to(DEVICE)

model.eval()

print()
print(f"Device : {DEVICE}")
print("Model mode : evaluation")


# ============================================================
# CHECK MODEL PARAMETERS
# ============================================================

print()
print("Model parameter check...")

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Total parameters : {total_parameters:,}"
)


# ============================================================
# CREATE TEST SEQUENCE
# ============================================================

print()
print("=" * 80)
print("CREATING TEST SEQUENCE")
print("=" * 80)

# ------------------------------------------------------------
# IMPORTANT
#
# This is ONLY a shape/inference test.
#
# The actual realtime system will replace this sequence
# with the real 5 transactions from customer history.
# ------------------------------------------------------------

test_sequence = torch.tensor(
    [
        [0.10] * NUM_FEATURES,
        [0.20] * NUM_FEATURES,
        [0.30] * NUM_FEATURES,
        [0.40] * NUM_FEATURES,
        [0.50] * NUM_FEATURES
    ],
    dtype=torch.float32
)

print()
print(f"Sequence shape : {tuple(test_sequence.shape)}")

print(
    f"Expected shape : ({SEQ_LEN}, {NUM_FEATURES})"
)


# ============================================================
# VALIDATE SEQUENCE
# ============================================================

if test_sequence.ndim != 2:

    raise ValueError(
        "Sequence must have 2 dimensions: "
        "(SEQ_LEN, NUM_FEATURES)"
    )


if test_sequence.shape[0] != SEQ_LEN:

    raise ValueError(
        f"Invalid sequence length. "
        f"Expected {SEQ_LEN}, "
        f"received {test_sequence.shape[0]}"
    )


if test_sequence.shape[1] != NUM_FEATURES:

    raise ValueError(
        f"Invalid number of features. "
        f"Expected {NUM_FEATURES}, "
        f"received {test_sequence.shape[1]}"
    )


print("Sequence shape validation : OK")


# ============================================================
# ADD BATCH DIMENSION
# ============================================================

print()
print("Adding batch dimension...")

x = test_sequence.unsqueeze(0)

# ------------------------------------------------------------
# Shape:
#
# (5, 15)
#     ↓
# (1, 5, 15)
# ------------------------------------------------------------

print(
    f"Input tensor shape : {tuple(x.shape)}"
)

print(
    "Expected tensor shape : "
    f"(1, {SEQ_LEN}, {NUM_FEATURES})"
)


# ============================================================
# MODEL INFERENCE
# ============================================================

print()
print("=" * 80)
print("RUNNING INFERENCE")
print("=" * 80)

with torch.no_grad():

    probability_tensor = model(
        x.to(DEVICE)
    )


# ============================================================
# OUTPUT
# ============================================================

probability = probability_tensor.item()

prediction = int(
    probability >= THRESHOLD
)


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 80)
print("INFERENCE RESULT")
print("=" * 80)

print()
print(
    f"Sequence length    : {SEQ_LEN}"
)

print(
    f"Number of features : {NUM_FEATURES}"
)

print(
    f"Input shape        : {tuple(x.shape)}"
)

print(
    f"Fraud probability  : {probability:.6f}"
)

print(
    f"Threshold          : {THRESHOLD:.2f}"
)

print(
    f"Prediction         : {prediction}"
)

if prediction == 1:

    print(
        "Result             : FRAUD"
    )

else:

    print(
        "Result             : LEGITIMATE"
    )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 80)
print("TEST COMPLETED SUCCESSFULLY")
print("=" * 80)

print()
print("Pipeline verified:")
print("  [OK] PyTorch")
print("  [OK] Checkpoint")
print("  [OK] Model architecture")
print("  [OK] model_state_dict")
print("  [OK] Input shape (5, 15)")
print("  [OK] LSTM")
print("  [OK] Attention")
print("  [OK] Fraud probability")
print("  [OK] Fraud prediction")
print()