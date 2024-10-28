import argparse


def get_config():
    parse = argparse.ArgumentParser(description='common train config')
    # parameters for the model
    parse.add_argument('-device', '--device_topofallfeature', type=str, nargs='?', default="cuda:2",
                       help="setting the cuda device")
    parse.add_argument('-dataset', '--dataset_topofallfeature', type=str, nargs='?', default="DrugVirus",
                       help="setting the dataset:MDAD, aBiofilm or DrugVirus ")
    config = parse.parse_args()
    return config
