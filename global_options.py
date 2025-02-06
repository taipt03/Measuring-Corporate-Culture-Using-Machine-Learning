import datetime
import itertools
import os
import gc
from multiprocessing import Pool
from pathlib import Path
from stanfordnlp.server import CoreNLPClient
import global_options
from culture import file_util, preprocess_parallel

def process_batch(batch_data):
    """Process a batch of documents.
    
    Args:
        batch_data: Tuple of (documents, document_ids)
    Returns:
        Tuple of (processed_lines, processed_ids)
    """
    documents, doc_ids = batch_data
    output_lines = []
    output_line_ids = []
    
    for doc, doc_id in zip(documents, doc_ids):
        try:
            output_line, output_line_id = preprocess_parallel.process_document(doc, doc_id)
            output_lines.append(output_line)
            output_line_ids.append(output_line_id)
        except Exception as e:
            print(f"Error processing document {doc_id}: {str(e)}")
            output_lines.append("")
            output_line_ids.append(doc_id)
    
    return output_lines, output_line_ids

def write_outputs(output_file, output_index_file, lines, line_ids):
    """Write processed outputs to files."""
    if lines:
        output_lines = "\n".join(lines) + "\n"
        output_line_ids = "\n".join(line_ids) + "\n"
        
        with open(output_file, "a", newline="\n") as f_out:
            f_out.write(output_lines)
        if output_index_file is not None:
            with open(output_index_file, "a", newline="\n") as f_out:
                f_out.write(output_line_ids)

def process_largefile(
    input_file,
    output_file,
    input_file_ids,
    output_index_file,
    function_name,
    chunk_size=100,
    start_index=None,
    batch_size=5  # Added batch_size parameter
):
    """Process large file with improved memory management.
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
        input_file_ids: List of input file IDs
        output_index_file: Path to output index file
        function_name: Processing function
        chunk_size: Size of chunks to process
        start_index: Starting index
        batch_size: Size of batches for parallel processing
    """
    # Clean up existing files if starting from beginning
    try:
        if start_index is None:
            os.remove(str(output_file))
            os.remove(str(output_index_file))
    except OSError:
        pass

    assert file_util.line_counter(input_file) == len(input_file_ids), \
        "Input file and ID file must have same number of rows."

    # Create a single pool for all processing
    with Pool(global_options.N_CORES) as pool:
        with open(input_file, newline="\n", encoding="utf-8", errors="ignore") as f_in:
            line_i = 0
            
            # Handle start index
            if start_index is not None:
                for _ in range(start_index):
                    next(f_in)
                input_file_ids = input_file_ids[start_index:]
                line_i = start_index

            # Process in chunks
            for next_n_lines, next_n_line_ids in zip(
                itertools.zip_longest(*[f_in] * chunk_size),
                itertools.zip_longest(*[iter(input_file_ids)] * chunk_size),
            ):
                line_i += chunk_size
                print(f"{datetime.datetime.now()} - Processing line: {line_i}")

                # Filter out None values
                next_n_lines = list(filter(None.__ne__, next_n_lines))
                next_n_line_ids = list(filter(None.__ne__, next_n_line_ids))

                # Split into smaller batches for better memory management
                for i in range(0, len(next_n_lines), batch_size):
                    batch_lines = next_n_lines[i:i + batch_size]
                    batch_ids = next_n_line_ids[i:i + batch_size]
                    
                    # Process batch
                    results = pool.map(process_batch, [(batch_lines, batch_ids)])
                    
                    # Write results
                    for output_lines, output_line_ids in results:
                        write_outputs(output_file, output_index_file, 
                                    output_lines, output_line_ids)
                    
                    # Force garbage collection
                    gc.collect()

def main():
    """Main function to run the processing pipeline."""
    corenlp_props = {
        "ner.applyFineGrained": "false",
        "annotators": "tokenize, ssplit, pos, lemma, ner, depparse",
        "timeout": "30000",  # Added timeout per document
        "pos.model": "edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger",
    }

    with CoreNLPClient(
        properties=corenlp_props,
        memory=global_options.RAM_CORENLP,
        threads=global_options.N_CORES,
        timeout=12000000,
        endpoint="http://localhost:9002",
        max_char_length=100000,  # Reduced from 1000000
        be_quiet=True  # Reduce logging
    ) as client:
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
        
        try:
            process_largefile(
                input_file=in_file,
                output_file=out_file,
                input_file_ids=in_file_index,
                output_index_file=output_index_file,
                function_name=preprocess_parallel.process_document,
                chunk_size=global_options.PARSE_CHUNK_SIZE,
                batch_size=5  # Process 5 documents at a time
            )
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            raise

if __name__ == "__main__":
    main()