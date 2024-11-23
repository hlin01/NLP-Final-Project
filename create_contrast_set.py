from datasets import load_dataset
import google.generativeai as genai
from tqdm import tqdm
import time
import json

# Configure Gemini API
genai.configure(api_key="API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

# Load and filter SNLI dataset
dataset = load_dataset("stanfordnlp/snli", split="test")
entailment_examples = dataset.filter(lambda x: x["label"] == 0).select(range(100))
contradiction_examples = dataset.filter(lambda x: x["label"] == 2).select(range(100))

# Prompt templates
ENTAILMENT_TO_CONTRADICTION_PROMPT = """
Given a premise and hypothesis pair where the hypothesis currently entails the premise, modify the hypothesis to contradict the premise while keeping the same basic topic and making minimal changes. Aim for plausible contradictions.

Premise: {premise}
Original hypothesis (entails): {hypothesis}

Output only the new contradicting hypothesis with no other text.
"""

CONTRADICTION_TO_ENTAILMENT_PROMPT = """
Given a premise and hypothesis pair where the hypothesis currently contradicts the premise, modify the hypothesis to entail the premise while keeping the same basic topic and making minimal changes. Ensure the modified hypothesis logically follows from the premise.

Premise: {premise}
Original hypothesis (contradicts): {hypothesis}

Output only the new entailing hypothesis with no other text.
"""

# Generate contrast set
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

# Generate the contrast sets
entailment_to_contradiction_set = generate_contrast_set(
    entailment_examples,
    ENTAILMENT_TO_CONTRADICTION_PROMPT,
    "entailment",
    "contradiction"
)

contradiction_to_entailment_set = generate_contrast_set(
    contradiction_examples,
    CONTRADICTION_TO_ENTAILMENT_PROMPT,
    "contradiction",
    "entailment"
)

# Save to JSON
with open("entailment_to_contradiction.json", "w") as f:
    json.dump(entailment_to_contradiction_set, f, indent=4)

with open("contradiction_to_entailment.json", "w") as f:
    json.dump(contradiction_to_entailment_set, f, indent=4)

# Modify prompt
# Set temperature and top_p
