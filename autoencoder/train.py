from autoencoder_trainer import AutoEncoderTrainer
from conv_autoencoder import ConvAutoEncoder
from slot_attn_autoencoder import SlotAttentionAutoEncoderV1, SlotAttentionAutoEncoderV2, SlotAttentionAutoEncoderv5
# from variational_autoencoder import VCAE
from dataset_loader import DatasetLoader

from config import *

# model = ConvAutoEncoder(
#     input_channel_size=3,
#     hidden_channel_sizes=HIDDEN_CHANNEL_SIZES, 
# )
# model = SlotAttentionAutoEncoderV1(
#     inchannels=3,
#     spatial_size=INPUT_SIZE[:2],
#     n_slots=8,
#     n_iters=3,
# )
model = SlotAttentionAutoEncoderv5(
    inchannels=3,
    hidden_channel_sizes=HIDDEN_CHANNEL_SIZES,
    n_slots=8,
    n_iters=3,
)

def train(args):
    train_dataloader = DatasetLoader(DATASET_PATH)

    trainer = AutoEncoderTrainer(
        model=model,
        dataset=train_dataloader, 
        batch_size=BATCH_SIZE,
        model_name=args.model_name
    )
    trainer.train()

def test(args):
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