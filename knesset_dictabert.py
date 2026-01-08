import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
import os
import sys

if __name__ == "__main__":
    # Check if the correct number of arguments was provided
    if len(sys.argv) != 3:
        print("Usage: python knesset_dictabert.py <input_file_path> <output_dir_path>")
        sys.exit(1)

    # Get paths from command line arguments
    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    # Define the output file name as required by instructions
    output_file = os.path.join(output_dir, "dictabert_results.txt")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)


    # Define models to try: DictaBERT first, then AlephBERT as a fallback [cite: 87]
    models_to_try = ["dicta-il/dictabert", "onlplab/alephbert-base"]
    mask_filler = None

    for model_id in models_to_try:
        try:
            # Load tokenizer and model with trust_remote_code=True for Dicta models [cite: 87]
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)

            # Create the Masked Language Modeling pipeline
            mask_filler = pipeline(task="fill-mask", model=model, tokenizer=tokenizer)
            break  # Exit loop if model loads successfully
        except Exception as e:
            print(f"Could not load {model_id}. Error: {e}")

    if mask_filler is None:
        print("\nCRITICAL ERROR: Could not load any Hebrew BERT model.")
    else:
        if not os.path.exists(input_file):
            print(f"Error: {input_file} not found!")
        else:
            output_content = []
            with open(input_file, 'r', encoding='utf-8') as f:
                # Read all non-empty lines from the input file
                lines = [line.strip() for line in f if line.strip()]

            for line in lines:
                # Replace the placeholder [*] with the standard BERT [MASK] token [cite: 89]
                formatted_input = line.replace("[*]", "[MASK]")

                try:
                    # Get the most probable prediction (top_k=1)
                    predictions = mask_filler(formatted_input, top_k=1)

                    # Handle cases where the model returns a nested list
                    if isinstance(predictions[0], list):
                        best_pred = predictions[0][0]
                    else:
                        best_pred = predictions[0]

                    # Extract the predicted token and the completed sequence [cite: 92]
                    predicted_token = best_pred['token_str']
                    completed_sentence = best_pred['sequence']

                    # Format output exactly as required by the report instructions [cite: 92]
                    output_content.append(f"masked_sentence: {formatted_input}")
                    output_content.append(f"dictaBERT_sentence: {completed_sentence}")
                    output_content.append(f"dictaBERT tokens: {predicted_token}")


                except Exception as e:
                    print(f"Error processing line '{line}': {e}")

            # Write final results to the output file [cite: 92]
            with open(output_file, 'w', encoding='utf-8') as f_out:
                f_out.write("\n".join(output_content))