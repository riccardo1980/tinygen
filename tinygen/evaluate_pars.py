from typing import Dict


class EvaluateParameters(object):
    model_path: str
    dataset_path: str
    num_classes: int
    shuffle_buffer_size: int
    batch_size: int

    def __init__(self, params: Dict) -> None:
        configs = {
            "model_path": params.pop("model_path"),
            "dataset_path": params.pop("dataset_path"),
            "num_classes": params.pop("num_classes"),
            "shuffle_buffer_size": params.pop("shuffle_buffer_size"),
            "batch_size": params.pop("batch_size"),
        }
        self.__dict__.update(configs)

    def __repr__(self) -> str:
        return str(self.__dict__)


def build_parameters(args: Dict) -> EvaluateParameters:
    pars = EvaluateParameters(args)
    return pars
