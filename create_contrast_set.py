from datasets import load_dataset
import google.generativeai as genai
from tqdm import tqdm
import time
import json

# Configure Gemini API
genai.configure(api_key="API_KEY")  # Replace with your API key
model = genai.GenerativeModel("gemini-1.5-pro")  # Replace with desired model

# Load and filter SNLI dataset
dataset = load_dataset("stanfordnlp/snli", split="test")

# Filter dataset by label
entailment_examples = dataset.filter(lambda x: x["label"] == 0).shuffle(seed=42).select(range(500))
neutral_examples = dataset.filter(lambda x: x["label"] == 1).shuffle(seed=42).select(range(500))
contradiction_examples = dataset.filter(lambda x: x["label"] == 2).shuffle(seed=42).select(range(500))

# Prompt templates
ENTAILMENT_TO_CONTRADICTION_PROMPT = """
Modify the hypothesis so that it directly contradicts the premise. Make the minimal necessary changes to create an explicit contradiction, ensuring the topic and language deviate as little as possible from the original. The contradiction must be obvious and leave no room for ambiguity.

Premise: {premise}
Original hypothesis (entails): {hypothesis}

Provide only the revised hypothesis that contradicts the premise.
"""

ENTAILMENT_TO_NEUTRAL_PROMPT = """
Modify the hypothesis so that it neither logically follows from nor contradicts the premise. The revised hypothesis can deviate from the original as long as it is relevant, plausible, and achieves neutrality.

Premise: {premise}
Original hypothesis (entails): {hypothesis}

Provide only the revised hypothesis that is neutral with respect to the premise.
"""

NEUTRAL_TO_ENTAILMENT_PROMPT = """
Modify the hypothesis so that it logically follows from the premise. The revised hypothesis can deviate from the original as long as it is relevant, plausible, and achieves a clear, consistent, and unambiguous entailment.

Premise: {premise}
Original hypothesis (neutral): {hypothesis}

Provide only the revised hypothesis that entails the premise.
"""

NEUTRAL_TO_CONTRADICTION_PROMPT = """
Modify the hypothesis so that it directly contradicts the premise. The revised hypothesis can deviate from the original as long as it is relevant, plausible, and the contradiction is explicit and leaves no ambiguity.

Premise: {premise}
Original hypothesis (neutral): {hypothesis}

Provide only the revised hypothesis that contradicts the premise.
"""

CONTRADICTION_TO_ENTAILMENT_PROMPT = """
Modify the hypothesis so that it logically follows from the premise. Make the minimal necessary changes to create a clear entailment, ensuring the revised hypothesis deviates as little as possible from the original while aligning fully with the premise.

Premise: {premise}
Original hypothesis (contradicts): {hypothesis}

Provide only the revised hypothesis that entails the premise.
"""

CONTRADICTION_TO_NEUTRAL_PROMPT = """
Modify the hypothesis so that it neither contradicts nor logically follows from the premise. The revised hypothesis can deviate from the original as long as it is relevant, plausible, and achieves neutrality.

Premise: {premise}
Original hypothesis (contradicts): {hypothesis}

Provide only the revised hypothesis that is neutral with respect to the premise.
"""

# Function to generate contrast sets
def generate_contrast_set(examples, prompt_template, original_label, new_label):
    contrast_set = []
    for example in tqdm(examples):
        try:
            premise = example["premise"]
            hypothesis = example["hypothesis"]
            prompt = prompt_template.format(premise=premise, hypothesis=hypothesis)

            # Generate the new hypothesis
            response = model.generate_content(prompt)
            perturbed_hypothesis = response.text.strip()

            # Add to contrast set
            contrast_set.append({
                "premise": premise,
                "original_hypothesis": hypothesis,
                "perturbed_hypothesis": perturbed_hypothesis,
                "original_label": original_label,
                "new_label": new_label
            })

            # Sleep to avoid rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"Error processing example: {e}")
            continue
    return contrast_set

# Generate all contrast sets
entailment_to_contradiction_set = generate_contrast_set(
    entailment_examples,
    ENTAILMENT_TO_CONTRADICTION_PROMPT,
    "entailment",
    "contradiction"
)

entailment_to_neutral_set = generate_contrast_set(
    entailment_examples,
    ENTAILMENT_TO_NEUTRAL_PROMPT,
    "entailment",
    "neutral"
)

neutral_to_entailment_set = generate_contrast_set(
    neutral_examples,
    NEUTRAL_TO_ENTAILMENT_PROMPT,
    "neutral",
    "entailment"
)

neutral_to_contradiction_set = generate_contrast_set(
    neutral_examples,
    NEUTRAL_TO_CONTRADICTION_PROMPT,
    "neutral",
    "contradiction"
)

contradiction_to_entailment_set = generate_contrast_set(
    contradiction_examples,
    CONTRADICTION_TO_ENTAILMENT_PROMPT,
    "contradiction",
    "entailment"
)

contradiction_to_neutral_set = generate_contrast_set(
    contradiction_examples,
    CONTRADICTION_TO_NEUTRAL_PROMPT,
    "contradiction",
    "neutral"
)

# Save all sets to JSON files
with open("entailment_to_contradiction.json", "w") as f:
    json.dump(entailment_to_contradiction_set, f, indent=4)

with open("entailment_to_neutral.json", "w") as f:
    json.dump(entailment_to_neutral_set, f, indent=4)

with open("neutral_to_entailment.json", "w") as f:
    json.dump(neutral_to_entailment_set, f, indent=4)

with open("neutral_to_contradiction.json", "w") as f:
    json.dump(neutral_to_contradiction_set, f, indent=4)

with open("contradiction_to_entailment.json", "w") as f:
    json.dump(contradiction_to_entailment_set, f, indent=4)

with open("contradiction_to_neutral.json", "w") as f:
    json.dump(contradiction_to_neutral_set, f, indent=4)
