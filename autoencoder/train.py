from autoencoder_trainer import AutoEncoderTrainer
from conv_autoencoder import ConvAutoEncoder
from variational_autoencoder import VCAE
from dataset_loader import DatasetLoader

from config import *

def train(args):
    model = ConvAutoEncoder(
        input_channel_size=3,
        hidden_channel_sizes=HIDDEN_CHANNEL_SIZES, 
    )
    # model = VCAE(
    #     input_channel_size=3,
    #     hidden_channel_sizes=HIDDEN_CHANNEL_SIZES,
    #     latent_size=20
    # )
    train_dataloader = DatasetLoader(DATASET_PATH)

    trainer = AutoEncoderTrainer(
        model=model,
        dataset=train_dataloader, 
        batch_size=BATCH_SIZE,
        model_name=args.model_name
    )
    trainer.train()

def test(args):
    model = ConvAutoEncoder(
        input_channel_size=3,
        hidden_channel_sizes=HIDDEN_CHANNEL_SIZES, 
    )
    dataset = DatasetLoader(DATASET_PATH, is_train=False)
    trainer = AutoEncoderTrainer(
        model=model,
        dataset=dataset, 
        batch_size=1,
        model_name=args.model_name
    )
    trainer.eval()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, help="Name of the model", default="default-model")
    parser.add_argument('-eval', '--eval', action="store_true")
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        test(args)