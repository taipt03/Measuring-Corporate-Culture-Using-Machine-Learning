import datetime
import itertools
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

import spacy
from transformers import pipeline

import global_options
from culture import file_util, preprocess


class CorpusPreprocessor:
    def __init__(self):
        # Load the NER model and spaCy for lemmatization
        self.ner_pipeline = pipeline("ner", device=0)  # Use GPU if available
        self.nlp = spacy.load("en_core_web_sm")  # Load spaCy model for lemmatization

    def process_document(self, doc, doc_id=None):
        """Annotate a document using Hugging Face NER model."""
        sentences = doc.split('. ')
        
        # Perform NER in batches
        batch_size = 16  # Adjustable based on hardware and benchmarking
        ner_results = self.ner_pipeline(sentences, batch_size=batch_size)
        
        # Lemmatize sentences in batches
        lemmatized_sentences = [
            " ".join([token.lemma_ for token in self.nlp(sentence)])
            for sentence in sentences
        ]

        sentences_processed = []
        doc_ids = []
        for i, (sentence, ner_result) in enumerate(zip(lemmatized_sentences, ner_results)):
            # Tagging NER results
            for entity in ner_result:
                start = entity['start']
                end = entity['end']
                label = entity['entity']
                sentence = sentence[:start] + f"[NER:{label}]" + sentence[start:end] + sentence[end:]
            sentences_processed.append(sentence)
            doc_ids.append(f"{doc_id}_{i}")
        return sentences_processed, doc_ids


def process_line(line, lineID):
    """Process each line and return a tuple of sentences and sentence IDs."""
    try:
        sentences_processed, doc_sent_ids = corpus_preprocessor.process_document(line, lineID)
    except Exception as e:
        print(e)
        print(f"Exception in line: {lineID}")
        return "", ""
    return "\n".join(sentences_processed), "\n".join(doc_sent_ids)


def process_chunk(args):
    """Process a chunk of lines."""
    next_n_lines, next_n_line_ids, function_name = args
    output_lines = []
    output_line_ids = []
    for output_line, output_line_id in map(function_name, next_n_lines, next_n_line_ids):
        output_lines.append(output_line)
        output_line_ids.append(output_line_id)
    return "\n".join(output_lines) + "\n", "\n".join(output_line_ids) + "\n"


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
    assert file_util.line_counter(input_file) == len(input_file_ids), "Input file must match the number of IDs."

    with open(input_file, newline="\n", encoding="utf-8", errors="ignore") as f_in:
        line_i = 0
        if start_index is not None:
            for _ in range(start_index):
                next(f_in)
            input_file_ids = input_file_ids[start_index:]
            line_i = start_index
        
        pool = Pool(cpu_count())  # Use all available CPU cores
        chunks = []
        
        for next_n_lines, next_n_line_ids in zip(
            itertools.zip_longest(*[f_in] * chunk_size),
            itertools.zip_longest(*[iter(input_file_ids)] * chunk_size),
        ):
            next_n_lines = list(filter(None.__ne__, next_n_lines))
            next_n_line_ids = list(filter(None.__ne__, next_n_line_ids))
            chunks.append((next_n_lines, next_n_line_ids, function_name))

        results = pool.map(process_chunk, chunks)  # Process chunks in parallel
        pool.close()
        pool.join()

        # Write results to output files
        for output_lines, output_line_ids in results:
            with open(output_file, "a", newline="\n") as f_out:
                f_out.write(output_lines)
            if output_index_file is not None:
                with open(output_index_file, "a", newline="\n") as f_out:
                    f_out.write(output_line_ids)


if __name__ == "__main__":
    # Initialize the corpus preprocessor
    corpus_preprocessor = CorpusPreprocessor()

    # Input and output file paths
    in_file = Path(global_options.DATA_FOLDER, "input", "documents.txt")
    in_file_index = file_util.file_to_list(
        Path(global_options.DATA_FOLDER, "input", "document_ids.txt")
    )
    out_file = Path(
        global_options.DATA_FOLDER, "processed", "parsed", "documents.txt"
    )
    output_index_file = Path(
        global_options.DATA_FOLDER, "processed", "parsed", "document_sent_ids.txt"
    )

    # Process the large file with batch processing
    process_largefile(
        input_file=in_file,
        output_file=out_file,
        input_file_ids=in_file_index,
        output_index_file=output_index_file,
        function_name=process_line,
        chunk_size=global_options.PARSE_CHUNK_SIZE,
    )
