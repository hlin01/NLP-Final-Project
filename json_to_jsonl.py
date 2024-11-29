import json

def convert_json_to_jsonl(input_file, output_file):
    # Mapping for labels
    label_mapping = {
        "neutral": 1,
        "entailment": 0,
        "contradiction": 2
    }

    # Read the input JSON file
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    # Prepare JSONL data
    jsonl_data = []
    for item in data:
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

# Example usage
input_file = 'contradiction_to_neutral.json'  # Replace with your input file name
output_file = 'contradiction_to_neutral.jsonl'  # Replace with your desired output file name
convert_json_to_jsonl(input_file, output_file)
