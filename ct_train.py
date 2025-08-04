from src.PreProcess import get_data_generators
from src.ct_model import build_ct_model
from src.train import train_model, plot_history
from src.config import CT_DATA_PATH, CT_MODEL_PATH, EPOCHS

# Load and preprocess data
train_gen, val_gen = get_data_generators(CT_DATA_PATH, augment=True)

# Build model
model = build_ct_model()

# Train model
history = train_model(model, train_gen, val_gen, CT_MODEL_PATH, EPOCHS)

# Plot results
plot_history(history)
