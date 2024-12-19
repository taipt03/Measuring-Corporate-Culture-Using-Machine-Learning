import os
import itertools
import logging
from tqdm import tqdm
from transformers import pipeline
import spacy

class CorpusPreprocessor:
    def __init__(self):
        # Load the NER model for both GPUs
        self.ner_pipeline_0 = pipeline("ner", device=0, aggregation_strategy="simple")  # Use GPU 0
        self.ner_pipeline_1 = pipeline("ner", device=1, aggregation_strategy="simple")  # Use GPU 1
        self.nlp = spacy.load("en_core_web_sm")  # Load spaCy model for lemmatization

    def process_document(self, doc, doc_id=None):
        """Main method: Annotate a document using Hugging Face NER model."""
        sentences = doc.split('. ')
        sentences_processed = []
        doc_ids = []
        
        # Batch processing
        batch_size = 32  # Adjust batch size as needed
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_ids = [f"{doc_id}_{j}" for j in range(i, min(i + batch_size, len(sentences)))]
            processed_batch = self.process_batch(batch)
            sentences_processed.extend(processed_batch)
            doc_ids.extend(batch_ids)

        return sentences_processed, doc_ids

    def process_batch(self, batch):
        """Process a batch of sentences."""
        ner_results = []
        for i, sentence in enumerate(batch):
            if i % 2 == 0:
                ner_results.append(self.ner_pipeline_0(sentence))
            else:
                ner_results.append(self.ner_pipeline_1(sentence))

        processed_sentences = []
        for i, sentence in enumerate(batch):
            doc = self.nlp(sentence)
            lemmatized_sentence = " ".join([token.lemma_ for token in doc])
            
            # Check if there are results before processing
            for entity in ner_results[i]:
                if 'entity' in entity:  # Check if 'entity' key exists
                    start = entity['start']
                    end = entity['end']
                    label = entity['entity']
                    lemmatized_sentence = (
                        lemmatized_sentence[:start] + f"[NER:{label}]" +
                        lemmatized_sentence[start:end] + lemmatized_sentence[end:]
                    )

            processed_sentences.append(lemmatized_sentence)

        return processed_sentences

def process_line(line, doc_id):
    """Process a single line of text."""
    preprocessor = CorpusPreprocessor()
    return preprocessor.process_document(line, doc_id)

def process_largefile(
    input_file,
    output_file,
    input_file_ids,
    output_index_file,
    function_name,
    chunk_size=100,
    start_index=None,
):
    """Transform an input file to processed documents and IDs."""
    try:
        if start_index is None:
            os.remove(str(output_file))
            os.remove(str(output_index_file))
    except OSError:
        pass

    total_lines = sum(1 for line in open(input_file, newline="\n", encoding="utf-8", errors="ignore"))
    
    with open(input_file, newline="\n", encoding="utf-8", errors="ignore") as f_in:
        line_i = 0
        if start_index is not None:
            for _ in range(start_index):
                next(f_in)
            input_file_ids = input_file_ids[start_index:]
            line_i = start_index
            
        with tqdm(total=total_lines, desc="Processing", unit="lines") as pbar:
            for next_n_lines, next_n_line_ids in zip(
                itertools.zip_longest(*[f_in] * chunk_size),
                itertools.zip_longest(*[iter(input_file_ids)] * chunk_size),
            ):
                line_i += chunk_size
                next_n_lines = list(filter(None.__ne__, next_n_lines))
                next_n_line_ids = list(filter(None.__ne__, next_n_line_ids))
                output_lines = []
                output_line_ids = []
                for output_line, output_line_id in map(function_name, next_n_lines, next_n_line_ids):
                    output_lines.append(output_line)
                    output_line_ids.append(output_line_id)

                output_lines = "\n".join(output_lines) + "\n"
                output_line_ids = "\n".join(output_line_ids) + "\n"
                
                with open(output_file, "a", newline="\n") as f_out:
                    f_out.write(output_lines)
                if output_index_file is not None:
                    with open(output_index_file, "a", newline="\n") as f_out:
                        f_out.write(output_line_ids)
                
                pbar.update(len(next_n_lines))  # Update the progress bar

# Example usage
if __name__ == "__main__":
    input_file = "documents.txt"
    output_file = "processed_documents.txt"
    input_file_ids = "document_ids.txt"
    output_index_file = "output_index.txt"
    
    # Read document IDs from file
    with open(input_file_ids, 'r') as f:
        document_ids = [line.strip() for line in f]
    
    process_largefile(input_file, output_file, document_ids, output_index_file, process_line)