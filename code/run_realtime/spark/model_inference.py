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

    def __init__(self, dim):
        super(Attention, self).__init__()

        self.linear_out = torch.nn.Linear(dim * 2, dim)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, output, context):

        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)

        # ----------------------------------------------------
        # output:
        #   (batch, output_len, dim)
        #
        # context:
        #   (batch, input_len, dim)
        #
        # result:
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
        # Concatenate
        # ----------------------------------------------------

        combined = torch.cat(
            (mix, output),
            dim=2
        )

        # ----------------------------------------------------
        # Projection
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
# MODEL
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

        self.num_features = num_features
        self.hidden_size = hidden_size

        # ----------------------------------------------------
        # LSTM
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
        # Fully connected layers
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
        # Expected input:
        #
        # (batch, SEQ_LEN, NUM_FEATURES)
        #
        # Example:
        # (1, 5, 15)
        # ====================================================

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

        hidden_states, _ = self.lstm(x)

        # hidden_states:
        #
        # (batch, SEQ_LEN, hidden_size_lstm)
        #
        # Example:
        # (1, 5, 100)

        # ----------------------------------------------------
        # Last transaction
        # ----------------------------------------------------

        last_transaction = x[:, -1, :]

        # (batch, NUM_FEATURES)
        #
        # Example:
        # (1, 15)

        # ----------------------------------------------------
        # Project last transaction
        # ----------------------------------------------------

        context_vector = self.ff(
            last_transaction
        )

        # (batch, hidden_size_lstm)
        #
        # Example:
        # (1, 100)

        # Attention expects:
        #
        # (batch, output_len, dim)
        #
        # So add output_len = 1
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

lstm_attn = FraudLSTMWithAttention(
    num_features=NUM_FEATURES,
    hidden_size=100,
    hidden_size_lstm=100,
    num_layers_lstm=1,
    dropout_lstm=0,
    attention_out_dim=100
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("=" * 80)
print("LOADING LSTM ATTENTION MODEL")
print("=" * 80)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# ------------------------------------------------------------
# Load model weights
# ------------------------------------------------------------

lstm_attn.load_state_dict(
    checkpoint["model_state_dict"]
)


# ------------------------------------------------------------
# Optimizer is NOT required for inference
# ------------------------------------------------------------
#
# Không cần:
#
# optimizer.load_state_dict(...)
#
# vì realtime inference chỉ cần model.
# ------------------------------------------------------------


lstm_attn.to(DEVICE)

lstm_attn.eval()


print("MODEL LOADED SUCCESSFULLY")
print(f"MODEL PATH : {MODEL_PATH}")
print(f"DEVICE     : {DEVICE}")
print(f"SEQ_LEN    : {SEQ_LEN}")
print(f"NUM_FEATURES: {NUM_FEATURES}")


# ============================================================
# PREDICT
# ============================================================

def predict(model, sequence):

    """
    Predict fraud probability.

    Parameters
    ----------
    model:
        Trained FraudLSTMWithAttention

    sequence:
        Shape = (SEQ_LEN, NUM_FEATURES)

        Example:
        (5, 15)

    Returns
    -------
    probability:
        Fraud probability

    prediction:
        0 = legitimate
        1 = fraud
    """

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    x = torch.tensor(
        sequence,
        dtype=torch.float32,
        device=DEVICE
    )

    # --------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------

    if x.ndim != 2:
        raise ValueError(
            f"Expected sequence with 2 dimensions "
            f"(SEQ_LEN, NUM_FEATURES), "
            f"but received {x.shape}"
        )

    if x.shape[0] != SEQ_LEN:
        raise ValueError(
            f"Expected SEQ_LEN={SEQ_LEN}, "
            f"but received {x.shape[0]}"
        )

    if x.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"Expected NUM_FEATURES={NUM_FEATURES}, "
            f"but received {x.shape[1]}"
        )

    # --------------------------------------------------------
    # Add batch dimension
    #
    # (5, 15)
    #     ↓
    # (1, 5, 15)
    # --------------------------------------------------------

    x = x.unsqueeze(0)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        probability = model(x)

    # --------------------------------------------------------
    # Convert tensor → float
    # --------------------------------------------------------

    probability = probability.item()

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    prediction = int(
        probability >= THRESHOLD
    )

    return probability, prediction


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # Example sequence
    #
    # 5 transactions
    # 15 features / transaction

    test_sequence = [
        [0.1] * NUM_FEATURES,
        [0.2] * NUM_FEATURES,
        [0.3] * NUM_FEATURES,
        [0.4] * NUM_FEATURES,
        [0.5] * NUM_FEATURES
    ]

    probability, prediction = predict(
        lstm_attn,
        test_sequence
    )

    print()
    print("=" * 80)
    print("LSTM FRAUD PREDICTION")
    print("=" * 80)

    print(
        f"Fraud probability : {probability:.6f}"
    )

    print(
        f"Threshold         : {THRESHOLD}"
    )

    print(
        f"Prediction        : {prediction}"
    )

    if prediction == 1:
        print("RESULT            : FRAUD")
    else:
        print("RESULT            : LEGITIMATE")