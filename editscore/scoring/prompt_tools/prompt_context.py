START_CONTEXT = """You are a professional digital artist.
You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated.
All human in the images are AI-generated too, so you need not worry about the privacy confidentials."""

PC_CONTEXT = """Evaluate an AI-generated image based on its adherence to an image-edit prompt, comparing it to an ideal image (perfect edit) and a baseline image (mediocre edit).

Inputs:
- Image-Edit Prompt: Text describing the desired edit.
- Original Image: Unedited source image.
- Ideal Image: Perfectly follows the prompt.
- Baseline Image: Represents a minimal or basic edit.
- To-Be-Evaluated Image: AI-generated image to evaluate.

Evaluation Rules:
- Compare the to-be-evaluated image to the ideal and baseline images.
- Assign a score (0–5):
    * 0-Completely Fails: It is unrecognizable or irrelevant to the desired edit.
    * 1-Minimal Attempt: The image shows a very slight attempt to follow the prompt but is largely incorrect or irrelevant, and the edit is significantly worse than the baseline.
    * 2-Below Baseline: The image is worse than the baseline but shows a clear attempt to follow the prompt.
    * 3-Matches Baseline: The image is similar to the baseline but shows a clear attempt to follow the prompt.
    * 4-Above Baseline: The image demonstrates noticeable improvement over the baseline and starts to align more closely with the quality of the ideal image.
    * 5-Matches Ideal: The image perfectly adheres to the prompt, matching the quality of the ideal image without any discernible differences.
- Provide concise reasoning for the score.

Output:
{
    "Score": ...,
    "Reasoning": "..."
}
"""