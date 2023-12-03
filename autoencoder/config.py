# Autoencoder parameters
INPUT_SIZE = (64, 64, 3)
HIDDEN_CHANNEL_SIZES = [8, 16, 32, 64]


# Training parameters
NUM_EPOCHS = 1000
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
TRAIN_TEST_SPLIT = 0.9

DATASET_PATH = "../crafter-logs/images/"
SAVE_PATH = "results/"
