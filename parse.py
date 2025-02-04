import datetime
import itertools
import os
from pathlib import Path
from multiprocessing import Pool
from typing import List, Tuple
import gc

from stanfordnlp.server import CoreNLPClient

import global_options
from culture import file_util, preprocess

def process_chunk(chunk_data: Tuple[List[str], List[str], CoreNLPClient]) -> Tuple[List[str], List[str]]:
    """Process a chunk of documents in parallel
    
    Arguments:
        chunk_data: Tuple containing (lines, line_ids, nlp_client)
        
    Returns:
        Tuple of (processed_lines, processed_ids)
    """
    lines, line_ids, client = chunk_data
    corpus_preprocessor = preprocess.preprocessor(client)
    
    output_lines = []
    output_line_ids = []
    
    for line, line_id in zip(lines, line_ids):
        try:
            sentences_processed, doc_sent_ids = corpus_preprocessor.process_document(line, line_id)
            output_lines.append("\n".join(sentences_processed))
            output_line_ids.append("\n".join(doc_sent_ids))
        except Exception as e:
            print(f"Exception in line {line_id}: {e}")
            continue
            
    return output_lines, output_line_ids

def batch_writer(filepath: Path, batch: List[str]):
    """Write a batch of lines to a file"""
    with open(filepath, "a", newline="\n") as f:
        f.write("\n".join(batch) + "\n")

def process_largefile(
    input_file: Path,
    output_file: Path,
    input_file_ids: List[str],
    output_index_file: Path,
    chunk_size: int = 100,
    start_index: int = None,
    n_processes: int = None
):
    """Optimized version of the large file processor with parallel processing and memory management"""
    
    # Clean up existing files if starting fresh
    if start_index is None:
        for filepath in (output_file, output_index_file):
            try:
                os.remove(str(filepath))
            except OSError:
                pass

    # Validate input
    assert file_util.line_counter(input_file) == len(input_file_ids), \
        "Input file and ID file must have same number of rows"

    # Initialize multiprocessing pool
    n_processes = n_processes or max(1, os.cpu_count() - 1)
    
    # Create CoreNLP clients for each process
    nlp_clients = [
        CoreNLPClient(
            properties={
                "ner.applyFineGrained": "false",
                "annotators": "tokenize, ssplit, pos, lemma, ner, depparse",
            },
            memory=f"{int(global_options.RAM_CORENLP/n_processes)}",
            threads=max(1, int(global_options.N_CORES/n_processes)),
            timeout=12000000,
            max_char_length=1000000,
        ) for _ in range(n_processes)
    ]

    # Process in batches
    with Pool(processes=n_processes) as pool:
        with open(input_file, newline="\n", encoding="utf-8", errors="ignore") as f_in:
            # Skip to start index if specified
            if start_index:
                for _ in range(start_index):
                    next(f_in)
                input_file_ids = input_file_ids[start_index:]

            # Process chunks in parallel
            line_i = start_index or 0
            write_batch_size = chunk_size * 2  # Adjust based on memory constraints
            
            output_buffer_lines = []
            output_buffer_ids = []
            
            for next_n_lines, next_n_line_ids in zip(
                itertools.zip_longest(*[f_in] * chunk_size),
                itertools.zip_longest(*[iter(input_file_ids)] * chunk_size),
            ):
                # Clean up None values
                next_n_lines = list(filter(None.__ne__, next_n_lines))
                next_n_line_ids = list(filter(None.__ne__, next_n_line_ids))
                
                # Split data for parallel processing
                chunk_data = [(lines, ids, client) for lines, ids, client in 
                            zip(next_n_lines, next_n_line_ids, nlp_clients)]
                
                # Process chunks
                results = pool.map(process_chunk, chunk_data)
                
                # Collect results
                for proc_lines, proc_ids in results:
                    output_buffer_lines.extend(proc_lines)
                    output_buffer_ids.extend(proc_ids)
                
                # Write when buffer is full
                if len(output_buffer_lines) >= write_batch_size:
                    batch_writer(output_file, output_buffer_lines)
                    batch_writer(output_index_file, output_buffer_ids)
                    output_buffer_lines = []
                    output_buffer_ids = []
                    gc.collect()  # Force garbage collection
                
                line_i += chunk_size
                print(f"{datetime.datetime.now()} - Processed through line: {line_i}")
            
            # Write remaining buffer
            if output_buffer_lines:
                batch_writer(output_file, output_buffer_lines)
                batch_writer(output_index_file, output_buffer_ids)

    # Cleanup
    for client in nlp_clients:
        client.stop()
    gc.collect()

if __name__ == "__main__":
    in_file = Path(global_options.DATA_FOLDER, "input", "documents.txt")
    in_file_index = file_util.file_to_list(
        Path(global_options.DATA_FOLDER, "input", "document_ids.txt")
    )
    out_file = Path(global_options.DATA_FOLDER, "processed", "parsed", "documents.txt")
    output_index_file = Path(
        global_options.DATA_FOLDER, "processed", "parsed", "document_sent_ids.txt"
    )
    
    process_largefile(
        input_file=in_file,
        output_file=out_file,
        input_file_ids=in_file_index,
        output_index_file=output_index_file,
        chunk_size=global_options.PARSE_CHUNK_SIZE,
        n_processes=None  # Will use CPU count - 1 by default
    )