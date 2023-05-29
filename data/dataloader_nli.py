# Loading script for an NLI dataset.
import json
import datasets

logger = datasets.logging.get_logger(__name__)


_URL = "./"
_TRAINING_FILE = "nli_train.json"
_DEV_FILE = "nli_validation.json" 
_TEST_FILE = "nli_test.json"


class nliConfig(datasets.BuilderConfig):
    """ Builder config for an NLI dataset """

    def __init__(self, **kwargs):
        """BuilderConfig.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(nliConfig, self).__init__(**kwargs)


class nli(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        nliConfig(
            name="NLI dataset",
            version=datasets.Version("1.0.1"),
            description="NLI dataset",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "premise": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                    "label": datasets.features.ClassLabel
                        (names=
                    [
                        "entailment",
                        "neutral",
                        "contradiction"
                    ]
                    ),
                    "label_cls":datasets.Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "dev": f"{_URL}{_DEV_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            data_dict = json.load(f)
            for id_, article in enumerate(data_dict):
                original_id = article["id"]
                premise     = article["premise"]
                hypothesis  = article["hypothesis"]
                label       = article["label"]
                label_cls   = article["cls_label"]
                yield id_, {
                    "id":           original_id,
                    "premise":      premise,
                    "hypothesis":   hypothesis,
                    "label":        label,
                    "label_cls":    label_cls
                }
