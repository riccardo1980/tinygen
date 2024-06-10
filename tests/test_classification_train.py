from typing import List, Tuple

import pytest
import tensorflow as tf

from tinygen.models.base_model import BaseClassifier
from tinygen.train_pars import TrainParameters

data_2_classes = [
    ("a a a a", 0),
    ("a a a b", 1),
    ("b b b a", 0),
    ("b b b b", 1),
    ("c c c a", 0),
    ("c c c b", 1),
]

data_3_classes = [
    ("a a a a", 0),
    ("a a a b", 1),
    ("a a a c", 2),
    ("b b b a", 0),
    ("b b b b", 1),
    ("b b b c", 2),
    ("c c c a", 0),
    ("c c c b", 1),
    ("c c c c", 2),
    ("d d d a", 0),
    ("d d d b", 1),
    ("d d d c", 2),
]

# fmt: off
real = [
    ("Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...", 0),  # noqa: E501
    ("Ok lar... Joking wif u oni...", 0),
    ("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's", 1),  # noqa: E501
    ("U dun say so early hor... U c already then say...", 0),
    ("Nah I don't think he goes to usf, he lives around here though", 0),
    ("FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv", 1),  # noqa: E501
    ("Even my brother is not like to speak with me. They treat me like aids patent.", 0),  # noqa: E501
    ("As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune", 0),  # noqa: E501
    ("WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.", 1),  # noqa: E501
    ("Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030", 1),  # noqa: E501
    ("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.", 0),  # noqa: E501
    ("SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info", 1)  # noqa: E501
]
# fmt: on


@pytest.mark.parametrize("data", [data_2_classes, data_3_classes, real])
def test_classification(data: List[Tuple[str, int]]) -> None:
    # build dataset

    num_classes = max(map(lambda d: d[1], data)) + 1

    # build parameters
    pars = TrainParameters(
        {
            "train_dataset_path": "aa",
            "eval_dataset_path": "bb",
            "output_path": "cc",
            "num_classes": num_classes,
            "shuffle_buffer_size": 2 * len(data),
            "batch_size": len(data),
            "epochs": 30,
            "embedding_dim": 16,
            "learning_rate": 0.005,
            "dropout": 0.0,
        }
    )

    X, y = zip(*data)  # noqa: N806
    train_dataset = tf.data.Dataset.from_tensor_slices((list(X), list(y)))
    train_dataset = train_dataset.map(lambda x, y: (x, tf.one_hot(y, pars.num_classes)))
    train_dataset = train_dataset.shuffle(pars.shuffle_buffer_size)
    train_dataset = train_dataset.batch(pars.batch_size)

    # vectorization layer
    vectorize_layer = tf.keras.layers.TextVectorization(
        output_mode="int", standardize=None
    )
    vectorize_layer.adapt(train_dataset.unbatch().map(lambda text, lbl: text))

    # vectorize the dataset
    def vectorize_text(
        text: tf.Tensor, label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    train_dataset = train_dataset.map(vectorize_text)
    train_dataset = train_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)

    # build model
    model = BaseClassifier(
        input_dim=len(vectorize_layer.get_vocabulary()),
        embedding_dim=pars.embedding_dim,
        num_classes=pars.num_classes,
        dropout=pars.dropout,
    )

    # compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=pars.learning_rate),
    )

    # train
    _ = model.fit(train_dataset, verbose=2, epochs=pars.epochs)

    # evaluate
    out = model.evaluate(train_dataset, verbose=2)
    out_dict = {k: v for k, v in zip(model.metrics_names, out)}

    assert out_dict["categorical_accuracy"] == 1.0
    for ii in range(num_classes):
        assert out_dict[f"precision_{ii}_p"] == 1.0
        assert out_dict[f"recall_{ii}_p"] == 1.0
