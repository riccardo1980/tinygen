import os
from typing import Dict

from tinygen.io.utils import convert_to_fuse


class parameters(object):
    train_dataset_path: str
    eval_dataset_path: str
    num_classes: int
    shuffle_buffer_size: int
    batch_size: int
    epochs: int
    model_path: str
    logs_path: str
    checkpoints_path: str
    embedding_dim: int
    learning_rate: float
    dropout: float

    def __init__(self, params: Dict) -> None:
        # manage reformatting
        configs = {
            "train_dataset_path": convert_to_fuse(params.pop("train_dataset_path")),
            "eval_dataset_path": convert_to_fuse(params.pop("eval_dataset_path")),
            "num_classes": params.pop("num_classes"),
            "shuffle_buffer_size": params.pop("shuffle_buffer_size"),
            "batch_size": params.pop("batch_size"),
            "epochs": params.pop("epochs"),
            "embedding_dim": params.pop("embedding_dim"),
            "learning_rate": params.pop("learning_rate"),
            "dropout": params.pop("dropout"),
        }

        output_subfolders: Dict[str, str] = {}
        if params["output_path"]:
            fused_path = convert_to_fuse(params["output_path"])
            output_subfolders = {
                "model_path": os.path.join(fused_path, "model"),
                "logs_path": os.path.join(fused_path, "logs"),
                "checkpoints_path": os.path.join(fused_path, "checkpoints"),
            }
        else:
            # use environment variables
            def _assert_is_present(key: str) -> str:
                out: str = os.getenv(key, "")
                assert out != "", f"{key} is not set"
                return out

            output_subfolders = {
                "model_path": _assert_is_present("AIP_MODEL_DIR"),
                "logs_path": _assert_is_present("AIP_TENSORBOARD_LOG_DIR"),
                "checkpoints_path": _assert_is_present("AIP_CHECKPOINT_DIR"),
            }

        configs.update(output_subfolders)
        self.__dict__.update(configs)

    def __repr__(self) -> str:
        return str(self.__dict__)


def get_parameters(args: Dict) -> parameters:
    pars = parameters(args)
    return pars
