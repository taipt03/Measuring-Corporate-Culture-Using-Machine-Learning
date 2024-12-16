import os
import spacy

def process_folder_with_spacy(input_folder, output_folder):
    """
    Processes all .txt files in the input folder using SpaCy for NER tagging and saves the processed
    files to the output folder.

    Args:
        input_folder (str): Path to the folder containing .txt files to process.
        output_folder (str): Path to the folder where processed files will be saved.
    """
    # Load the SpaCy Transformer model (GPU-enabled)
    nlp = spacy.load("en_core_web_trf")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            input_file_path = os.path.join(input_folder, file_name)

            # Read the content of the file
            with open(input_file_path, "r", encoding="utf-8") as file:
                text = file.read()

            # Process the text using SpaCy
            doc = nlp(text)
            processed_sentences = []

            for sent in doc.sents:  # Iterate over sentences
                sentence = []
                for token in sent:  # Iterate over tokens
                    # Append the token with NER tagging if applicable
                    token_text = (
                        f"[NER:{token.ent_type_}]{token.text}" if token.ent_type_ else token.text
                    )
                    sentence.append(token_text)
                processed_sentences.append(" ".join(sentence))

            # Save processed sentences to the output folder
            output_file_path = os.path.join(output_folder, file_name)
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                for sentence in processed_sentences:
                    output_file.write(sentence + "\n")
            print(f"Processed and saved: {file_name}")

if __name__ == "__main__":
    # Define the paths
    input_folder = "path/to/input/folder"  # Replace with your input folder path
    output_folder = "path/to/output/folder"  # Replace with your output folder path

    # Process the folder
    process_folder_with_spacy(input_folder, output_folder)
    print("Processing complete!")
