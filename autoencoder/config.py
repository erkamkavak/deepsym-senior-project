# Autoencoder parameters
INPUT_SIZE = (64, 64, 3)
HIDDEN_CHANNEL_SIZES = [8, 16, 32, 64, 128]


# Training parameters
NUM_EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE = 0.001
TRAIN_TEST_SPLIT = 0.9

DATASET_PATH = "../crafter-logs/images/"
SAVE_PATH = "results/"
