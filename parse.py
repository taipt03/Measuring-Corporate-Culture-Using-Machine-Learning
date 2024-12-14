import os
import itertools
import datetime
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import spacy

# Initialize SpaCy for sentence splitting (using GPU if available)
spacy_model = "en_core_web_sm"  # Use "en_core_web_trf" for transformer-based sentence splitting
spacy.require_gpu()  # Enable GPU acceleration for SpaCy (if available)
nlp = spacy.load(spacy_model)

# Load Hugging Face model for NER
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, device=0)  # Use GPU if available

def process_document(document, document_id):
    """
    Process a single document, split into sentences, and apply NER.
    Args:
        document (str): The document text.
        document_id (str): The document ID.
    Returns:
        tuple: Processed sentences and their corresponding IDs.
    """
    try:
        doc = nlp(document)  # Sentence splitting using SpaCy
        sentences = [sent.text for sent in doc.sents]
        sentence_ids = [f"{document_id}_{i}" for i in range(len(sentences))]
        
        # Apply NER to each sentence
        ner_results = [ner_pipeline(sentence) for sentence in sentences]
        
        # Optionally, format NER results (can be extended for downstream tasks)
        formatted_sentences = [
            f"{sentence}\nEntities: {result}" for sentence, result in zip(sentences, ner_results)
        ]
        
        return "\n".join(formatted_sentences), "\n".join(sentence_ids)
    except Exception as e:
        print(f"Error processing document {document_id}: {e}")
        return "", ""

def process_largefile(input_file, output_file, input_ids, output_index_file, chunk_size=100, start_index=None):
    """
    Process a large input file line by line in chunks, apply NLP processing, and save results.
    Args:
        input_file (str or Path): Path to the input file (one document per line).
        output_file (str or Path): Path to save processed sentences.
        input_ids (list): List of document IDs corresponding to each line.
        output_index_file (str or Path): Path to save sentence IDs.
        chunk_size (int): Number of lines to process in each chunk.
        start_index (int, optional): Line index to resume from.
    """
    try:
        # Remove existing output files if starting fresh
        if start_index is None:
            os.remove(output_file)
            os.remove(output_index_file)
    except OSError:
        pass

    assert len(input_ids) == sum(1 for _ in open(input_file)), \
        "Input file and input ID file must have the same number of lines."

    with open(input_file, "r", encoding="utf-8") as f_in:
        # Skip to the start index if resuming
        if start_index is not None:
            for _ in range(start_index):
                next(f_in)
            input_ids = input_ids[start_index:]

        line_i = start_index or 0
        for chunk_lines, chunk_ids in zip(
            itertools.zip_longest(*[f_in] * chunk_size),
            itertools.zip_longest(*[iter(input_ids)] * chunk_size),
        ):
            chunk_lines = list(filter(None, chunk_lines))  # Remove None values
            chunk_ids = list(filter(None, chunk_ids))  # Remove None values
            line_i += len(chunk_lines)

            print(f"[{datetime.datetime.now()}] Processing lines {line_i - len(chunk_lines)}-{line_i}.")

            output_sentences, output_sentence_ids = [], []
            for line, doc_id in zip(chunk_lines, chunk_ids):
                sentences, sentence_ids = process_document(line.strip(), doc_id.strip())
                output_sentences.append(sentences)
                output_sentence_ids.append(sentence_ids)

            # Write results to output files
            with open(output_file, "a", encoding="utf-8") as f_out:
                f_out.write("\n".join(output_sentences) + "\n")
            with open(output_index_file, "a", encoding="utf-8") as f_out:
                f_out.write("\n".join(output_sentence_ids) + "\n")

if __name__ == "__main__":
    # File paths and configuration
    input_file_path = Path("data/input/documents.txt")
    input_ids_path = Path("data/input/document_ids.txt")
    output_file_path = Path("data/output/processed_documents.txt")
    output_index_path = Path("data/output/processed_document_ids.txt")

    # Load document IDs
    with open(input_ids_path, "r", encoding="utf-8") as f:
        document_ids = [line.strip() for line in f]

    # Process the input file
    process_largefile(
        input_file=input_file_path,
        output_file=output_file_path,
        input_ids=document_ids,
        output_index_file=output_index_path,
        chunk_size=50,  # Adjust based on system memory and GPU capability
    )
