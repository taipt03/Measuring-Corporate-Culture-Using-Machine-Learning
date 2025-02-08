import datetime
import multiprocessing
from pathlib import Path
from functools import partial
from typing import List, Tuple, Callable
import itertools
from stanfordnlp.server import CoreNLPClient

def process_chunk(function_name: Callable, docs_with_ids: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Process a chunk of documents in parallel"""
    results = []
    for doc, doc_id in docs_with_ids:
        output_line, output_line_id = function_name(doc, doc_id)
        if output_line and output_line_id:
            results.append((output_line, output_line_id))
    return results

def process_largefile_parallel(
    input_file: str,
    output_file: str,
    input_file_ids: List[str],
    output_index_file: str,
    function_name: Callable,
    n_processes: int = 4,
    chunk_size: int = 100,
    start_index: int = None,
    max_char_length: int = 1000000,
):
    """Parallel version of process_largefile that processes multiple documents simultaneously.
    
    Arguments:
        input_file: path to text file, each line is a document
        output_file: processed lines file
        input_file_ids: list of input line IDs
        output_index_file: path to index file of output
        function_name: function that processes strings and returns processed strings and IDs
        n_processes: number of parallel processes to use
        chunk_size: number of documents per chunk
        start_index: line number to start from (0-based)
        max_char_length: maximum character limit per line
    """
    # Clear output files if starting fresh
    if start_index is None:
        for file in [output_file, output_index_file]:
            try:
                Path(file).unlink()
            except FileNotFoundError:
                pass

    # Read input documents and IDs
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        if start_index:
            for _ in range(start_index):
                next(f)
            input_file_ids = input_file_ids[start_index:]
        
        # Process in chunks
        pool = multiprocessing.Pool(processes=n_processes)
        
        # Create chunks of documents with their IDs
        for chunk_start in range(0, len(input_file_ids), chunk_size):
            print(f"{datetime.datetime.now()} - Processing chunk starting at line {chunk_start}")
            
            # Get next chunk of documents and IDs
            chunk_docs = []
            chunk_size_actual = 0
            
            for _ in range(chunk_size):
                try:
                    line = next(f)
                    if len(line) > max_char_length:
                        print(f"Skipping line {input_file_ids[chunk_start + chunk_size_actual]}: exceeds max length")
                        continue
                    chunk_docs.append((line, input_file_ids[chunk_start + chunk_size_actual]))
                    chunk_size_actual += 1
                except StopIteration:
                    break
            
            if not chunk_docs:
                break
                
            # Split chunk into sub-chunks for parallel processing
            sub_chunk_size = max(1, len(chunk_docs) // n_processes)
            sub_chunks = [chunk_docs[i:i + sub_chunk_size] 
                         for i in range(0, len(chunk_docs), sub_chunk_size)]
            
            # Process sub-chunks in parallel
            partial_process = partial(process_chunk, function_name)
            results = pool.map(partial_process, sub_chunks)
            
            # Flatten results and write to files
            all_results = [item for sublist in results for item in sublist]
            
            if all_results:
                with open(output_file, 'a', encoding='utf-8', newline='\n') as f_out:
                    f_out.write('\n'.join(result[0] for result in all_results) + '\n')
                    
                with open(output_index_file, 'a', encoding='utf-8', newline='\n') as f_out:
                    f_out.write('\n'.join(result[1] for result in all_results) + '\n')
        
        pool.close()
        pool.join()

if __name__ == "__main__":
    # Example usage with CoreNLPClient
    with CoreNLPClient(
        properties={
            "ner.applyFineGrained": "false",
            "annotators": "tokenize, ssplit, pos, lemma, ner, depparse",
        },
        memory=global_options.RAM_CORENLP,
        threads=global_options.N_CORES,
        timeout=12000000,
        max_char_length=1000000,
    ) as client:
        corpus_preprocessor = preprocess.preprocessor(client)
        
        # File paths
        in_file = Path(global_options.DATA_FOLDER, "input", "documents.txt")
        in_file_index = file_util.file_to_list(
            Path(global_options.DATA_FOLDER, "input", "document_ids.txt")
        )
        out_file = Path(global_options.DATA_FOLDER, "processed", "parsed", "documents.txt")
        output_index_file = Path(
            global_options.DATA_FOLDER, "processed", "parsed", "document_sent_ids.txt"
        )
        
        # Process files in parallel
        process_largefile_parallel(
            input_file=in_file,
            output_file=out_file,
            input_file_ids=in_file_index,
            output_index_file=output_index_file,
            function_name=process_line,
            n_processes=4,  # Number of parallel processes
            chunk_size=global_options.PARSE_CHUNK_SIZE,
        )