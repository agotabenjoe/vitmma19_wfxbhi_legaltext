# Configuration settings for LegalText classifier

# Training hyperparameters
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.03708204425095935
EMBED_DIM = 50
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_CLASSES = 5

# Paths (relative to project root /work in Docker)
DATA_DIR = "data"
DATASET_PATH = "data/final_split_dataset"
MODEL_DIR = "models"
BEST_MODEL_PATH = "models/best_final_model.pt"
BEST_MODEL_CONFIG_PATH = "models/best_final_model_config.json"
BEST_MODEL_VOCAB_PATH = "models/best_final_model_vocab.json"
BEST_ACC_PATH = "models/best_final_acc.txt"
