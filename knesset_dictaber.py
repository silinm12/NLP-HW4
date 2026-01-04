import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
import os


def run_step4_dictabert():
    print("--- Starting Step 4: DictaBERT Masked Language Modeling ---")

    # We will try DictaBERT first, then AlephBERT as a fallback
    models_to_try = ["dicta-nlp/dictabert", "onlplab/alephbert-base"]
    mask_filler = None

    for model_id in models_to_try:
        print(f"Attempting to load: {model_id}...")
        try:
            # Loading with trust_remote_code as required for Dicta models
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
            mask_filler = pipeline("fill-mask", model=model, tokenizer=tokenizer)
            print(f"Successfully loaded {model_id}!")
            break  # Exit the loop once a model is successfully loaded
        except Exception as e:
            print(f"Could not load {model_id}. Error: {e}")

    if mask_filler is None:
        print("\nCRITICAL ERROR: Could not load any Hebrew BERT model.")
        print("Please check your internet connection or try using a mobile hotspot.")
        return

    input_file = "masked_sampled_sents.txt"
    output_file = "results_dictabert.txt"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return

    output_content = []
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(lines)} sentences...")
    for line in lines:
        # Step A: Standardize the mask token
        formatted_input = line.replace("[*]", "[MASK]")

        try:
            # Step B: Get predictions
            predictions = mask_filler(formatted_input, top_k=1)

            # Step C: Fix the list-index error
            # If the model returns a nested list, we grab the first inner element
            if isinstance(predictions[0], list):
                best_pred = predictions[0][0]
            else:
                best_pred = predictions[0]

            # Step D: Extract the predicted word and full sentence
            predicted_token = best_pred['token_str']
            completed_sentence = best_pred['sequence']

            # Step E: Format exactly as the instructions require
            output_content.append(f"masked_sentence: {formatted_input}")
            output_content.append(f"dictaBERT_sentence: <{completed_sentence}>")
            output_content.append(f"dictaBERT tokens: {predicted_token}")
            output_content.append("")

            print(f"Successfully predicted: {predicted_token}")

        except Exception as e:
            print(f"Error processing line: {e}")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write("\n".join(output_content))

    print(f"Done! Results saved to {output_file}")


if __name__ == "__main__":
    run_step4_dictabert()