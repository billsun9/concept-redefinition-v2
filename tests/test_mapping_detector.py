import pandas as pd

from scripts.train_mapping_detector import split_specs


def test_mapping_detector_builds_strict_split_masks():
    rows = pd.DataFrame(
        [
            {
                "template_split": "train_template",
                "pair_split": "train_pair",
                "concept_category": "animals",
            },
            {
                "template_split": "test_template",
                "pair_split": "test_pair",
                "concept_category": "colors",
            },
        ]
    )
    specs = {
        (split_type, held_out): (train_mask, test_mask)
        for split_type, held_out, train_mask, test_mask in split_specs(rows)
    }

    template_train, template_test = specs[
        ("held_out_templates", "test_template")
    ]
    assert template_train.tolist() == [True, False]
    assert template_test.tolist() == [False, True]

    category_train, category_test = specs[
        ("held_out_category", "colors")
    ]
    assert category_train.tolist() == [True, False]
    assert category_test.tolist() == [False, True]
