from models import *
from trainers import *
from dataset_loader import DatasetLoader

from config import *

def train(model_name, model, dataset_path, TRAINER):
    train_dataloader = DatasetLoader(dataset_path)

    trainer = TRAINER(
        model=model,
        dataset=train_dataloader, 
        batch_size=BATCH_SIZE,
        model_name=model_name
    )
    trainer.train()
    # model.save_model_parts()

def test(model_name, model, dataset_path, TRAINER):
    test_dataloader = DatasetLoader(dataset_path, is_train=False)
    trainer = TRAINER(
        model=model,
        dataset=test_dataloader, 
        batch_size=1,
        model_name=model_name
    )
    trainer.eval()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--config-file", type=str, help="Model config file name", default="")
    parser.add_argument('-eval', '--eval', action="store_true")
    args = parser.parse_args()

    import yaml
    with open(f"train-configs/{args.config_file}.yaml", "r") as config_file: 
        try: 
            model_config = yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            print(exc)
    
    model_name = model_config.get("model_name")
    dataset_name = model_config.get("dataset_name")
    model = model_config.get("model", {})
    trainer = model_config.get("trainer", {})

    dataset_path = f"{DATASET_PATH}/{dataset_name}.hdf5"

    model_class_name = model.get('name')
    model_params = model.get('params', {})
    model_type = globals().get(model_class_name)
    if model_type is not None:
        model_instance = model_type(**model_params)
    else: 
        raise Exception(f"Error: Model class {model_class_name} not found.")

    trainer_class = globals().get(trainer)

    # model = ConvAutoEncoder(
    #     input_channel_size=3,
    #     hidden_channel_sizes=HIDDEN_CHANNEL_SIZES, 
    #     model_name=args.model_name
    # )
    # model = ConvBinaryFeatureAutoEncoder(
    #     input_channel_size=3, 
    #     hidden_channel_sizes=HIDDEN_CHANNEL_SIZES
    # )
    # model = SlotAttentionAutoEncoderV1(
    #     inchannels=3,
    #     spatial_size=INPUT_SIZE[:2],
    #     n_slots=8,
    #     n_iters=3,
    # )
    # model = SlotAttentionAutoEncoderv5(
    #     inchannels=3,
    #     hidden_channel_sizes=HIDDEN_CHANNEL_SIZES,
    #     n_slots=8,
    #     n_iters=3,
    # )
    if not args.eval:
        train(model_name, model_instance, dataset_path, trainer_class)
    else:
        test(model_name, model_instance, dataset_path, trainer_class)