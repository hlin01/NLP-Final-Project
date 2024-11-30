import json

def combine_and_convert_to_jsonl(input_files, output_file):
    """
    Combines multiple JSON files, transforms them into JSONL format, and writes the output to a file.
    
    Args:
    input_files (list of str): List of input JSON file paths.
    output_file (str): Path to the output JSONL file.
    """
    # Mapping for labels
    label_mapping = {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2
    }

    combined_data = []

    # Read and combine all input JSON files
    for input_file in input_files:
        with open(input_file, 'r') as infile:
            data = json.load(infile)
            combined_data.extend(data)

    # Prepare JSONL data
    jsonl_data = []
    for item in combined_data:
        jsonl_entry = {
            "premise": item["premise"],
            "hypothesis": item["perturbed_hypothesis"],
            "label": label_mapping.get(item["new_label"], -1)  # -1 for unmapped labels
        }
        jsonl_data.append(jsonl_entry)

    # Write the JSONL file
    with open(output_file, 'w') as outfile:
        for entry in jsonl_data:
            outfile.write(json.dumps(entry) + "\n")

input_files = [
    'entailment_to_neutral.json',
    'entailment_to_contradiction.json',
    'neutral_to_entailment.json',
    'neutral_to_contradiction.json',
    'contradiction_to_entailment.json',
    'contradiction_to_neutral.json'
]
output_file = 'contrast_set_2.jsonl'
combine_and_convert_to_jsonl(input_files, output_file)
