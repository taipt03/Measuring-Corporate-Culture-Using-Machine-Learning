import spacy
import os
from tqdm import tqdm

def process_document(doc, nlp):
    """
    Process a single document using SpaCy.

    Arguments:
        doc (str): The raw text of the document.
        nlp (spacy.Language): The SpaCy pipeline.

    Returns:
        list: A list of processed sentences with NER tags and dependencies.
    """
    processed_sentences = []
    spacy_doc = nlp(doc)  # Process document using SpaCy on GPU
    
    for sent in spacy_doc.sents:  # Iterate through sentences
        sentence_tokens = []
        for token in sent:
            # Token with lemma, POS, and NER (if applicable)
            token_data = f"{token.text}[lemma:{token.lemma_}][pos:{token.pos_}]"
            if token.ent_type_:
                token_data = f"[NER:{token.ent_type_}]{token_data}"
            sentence_tokens.append(token_data)
        processed_sentences.append(" ".join(sentence_tokens))
    
    return processed_sentences

def process_files_in_directory(input_dir, output_dir, nlp):
    """
    Process all files in a directory and save processed results.

    Arguments:
        input_dir (str): Directory containing input text files.
        output_dir (str): Directory to save processed files.
        nlp (spacy.Language): The SpaCy pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for file_name in tqdm(os.listdir(input_dir), desc="Processing files"):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, f"processed_{file_name}")
        
        # Skip non-text files
        if not file_name.endswith('.txt'):
            continue
        
        # Read document
        with open(input_path, 'r', encoding='utf-8') as file:
            doc = file.read()
        
        # Process document
        processed_sentences = process_document(doc, nlp)
        
        # Write processed document
        with open(output_path, 'w', encoding='utf-8') as output_file:
            for sentence in processed_sentences:
                output_file.write(sentence + '\n')

if __name__ == "__main__":
    # Use GPU if available
    spacy.require_gpu()

    # Load the SpaCy pipeline with the transformer model
    nlp = spacy.load("en_core_web_trf")  # GPU-enabled transformer model

    # Input and output directories
    input_directory = "/kaggle/input/non-annual-text/dictionary-based"
    output_directory = "/kaggle/working//documents"

    # Process files
    process_files_in_directory(input_directory, output_directory, nlp)
