# Loading script for the TECA dataset.
import json
import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """
            @inproceedings{armengol-estape-etal-2021-multilingual,
                title = "Are Multilingual Models the Best Choice for Moderately Under-resourced Languages? {A} Comprehensive Assessment for {C}atalan",
                author = "Armengol-Estap{\'e}, Jordi  and
                  Carrino, Casimiro Pio  and
                  Rodriguez-Penagos, Carlos  and
                  de Gibert Bonet, Ona  and
                  Armentano-Oller, Carme  and
                  Gonzalez-Agirre, Aitor  and
                  Melero, Maite  and
                  Villegas, Marta",
                booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
                month = aug,
                year = "2021",
                address = "Online",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/2021.findings-acl.437",
                doi = "10.18653/v1/2021.findings-acl.437",
                pages = "4933--4946",
            }
            """

_DESCRIPTION = """
               TECA consists of two subsets of textual entailment in Catalan, *catalan_TE1* and *vilaweb_TE*, which contain 14997 and 6166 pairs of premises and hypotheses, annotated according to the inference relation they have (implication, contradiction or neutral). This dataset was developed by BSC TeMU as part of the AINA project and intended as part of the Catalan Language Understanding Benchmark (CLUB).
               """

_HOMEPAGE = """https://zenodo.org/record/4621378"""

# TODO: upload datasets to github
_URL = "./"
_TRAINING_FILE = "nli_train.json"
_DEV_FILE = "nli_validation.json"  # new_dev_set
_TEST_FILE = "nli_test.json"


class tecaConfig(datasets.BuilderConfig):
    """ Builder config for the TECA dataset """

    def __init__(self, **kwargs):
        """BuilderConfig for TECA.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(tecaConfig, self).__init__(**kwargs)


class teca(datasets.GeneratorBasedBuilder):
    """ TECA Dataset """

    BUILDER_CONFIGS = [
        tecaConfig(
            name="teca",
            version=datasets.Version("1.0.1"),
            description="teca dataset",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
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
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
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
