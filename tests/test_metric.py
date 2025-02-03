##############################
### author : Ge Ya (Olga) Luo
##############################

import pytest
import os

def test_get_baseline_edit():
    from editscore.scoring.metric.editscores import EditScore
    model = EditScore(key_path="/home/mila/x/xuolga/keys/olga_personal_metric.env")
    output = model.evaluate(
                input_image='https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg',
                edit_instruction="Change the color of the zebra.",
                ideal_edit='https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_2.jpg',
                new_edit='https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_2.jpg',
                fs_input_images=['https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg'],
                fs_edit_instructions=["Change the color of the zebra."],
                fs_ideal_edits=['https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_2.jpg'],
                fs_new_edits=['https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_2.jpg'],
                fs_suggested_scores=[2],
                fs_reasonings=["The zebra's shape changed."],
            )
    assert isinstance(output, dict)
    assert "score" in output
    assert "reasoning" in output
    assert isinstance(output["score"], int) or isinstance(output["score"], list)
    assert isinstance(output["reasoning"], str)