from crafter_data_collector import CrafterDataCollector
from minerl_data_collector import MinerlDataCollector
from griddly_data_collector import GriddlyDataCollector
from enum import Enum


class DatasetEnvironment(Enum): 
    crafter = CrafterDataCollector
    minerl = MinerlDataCollector
    griddly = GriddlyDataCollector

    def __str__(self) -> str:
       return self.name


if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, help="Name of the dataset", default="default-dataset")
    parser.add_argument("--environment", type=lambda x: DatasetEnvironment[x], help="Name of the dataset", choices=list(DatasetEnvironment))
    args = parser.parse_args()

    data_collector_model = args.environment.value

    data_collector = data_collector_model(args.name)
    data_collector.collect_all()