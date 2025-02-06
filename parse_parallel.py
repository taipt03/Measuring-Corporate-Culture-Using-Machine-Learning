"""Implementation of parse.py that supports multiprocess
Main differences are 1) using Pool.starmap in process_largefile and 2) attach to local CoreNLP server in process_largefile.process_document
"""
import datetime
import itertools
import os
from multiprocessing import Pool
from pathlib import Path

from stanfordnlp.server import CoreNLPClient

import global_options
from culture import file_util, preprocess_parallel


def process_largefile(input_file, output_file, input_file_ids, output_index_file, function_name, chunk_size=100, start_index=None):
    try:
        if start_index is None:
            os.remove(str(output_file))
            os.remove(str(output_index_file))
    except OSError:
        pass
    
    assert file_util.line_counter(input_file) == len(input_file_ids), "Make sure the input file has the same number of rows as the input ID file."

    with open(input_file, newline="\n", encoding="utf-8", errors="ignore") as f_in:
        line_i = 0
        if start_index is not None:
            for _ in range(start_index):
                next(f_in)
            input_file_ids = input_file_ids[start_index:]
            line_i = start_index
            
        for next_n_lines, next_n_line_ids in zip(
            itertools.zip_longest(*[f_in] * chunk_size),
            itertools.zip_longest(*[iter(input_file_ids)] * chunk_size),
        ):
            try:
                line_i += chunk_size
                print(datetime.datetime.now())
                print(f"Processing line: {line_i}.")
                
                next_n_lines = list(filter(None.__ne__, next_n_lines))
                next_n_line_ids = list(filter(None.__ne__, next_n_line_ids))
                output_lines = []
                output_line_ids = []
                
                # Process chunk with new pool each time
                with Pool(global_options.N_CORES) as pool:
                    results = pool.starmap(
                        function_name, zip(next_n_lines, next_n_line_ids)
                    )
                    for output_line, output_line_id in results:
                        output_lines.append(output_line)
                        output_line_ids.append(output_line_id)
                
                # Write results
                if output_lines:  # Only process if we have output
                    output_text = "\n".join(output_lines) + "\n"
                    output_ids = "\n".join(output_line_ids) + "\n"
                    
                    with open(output_file, "a", newline="\n") as f_out:
                        f_out.write(output_text)
                    if output_index_file is not None:
                        with open(output_index_file, "a", newline="\n") as f_out:
                            f_out.write(output_ids)
                
                # Clean up variables
                if 'output_lines' in locals(): del output_lines
                if 'output_line_ids' in locals(): del output_line_ids
                if 'output_text' in locals(): del output_text
                if 'output_ids' in locals(): del output_ids
                if 'results' in locals(): del results
                
                # Force garbage collection
                import gc
                gc.collect()
                
            except Exception as e:
                print(f"Error processing chunk at line {line_i}: {str(e)}")
                raise e

if __name__ == "__main__":
    with CoreNLPClient(
        properties={
            "ner.applyFineGrained": "false",
            "annotators": "tokenize, ssplit, pos, lemma, ner, depparse",
        },
        memory=global_options.RAM_CORENLP,
        threads=global_options.N_CORES,
        timeout=12000000,
        endpoint="http://localhost:9002",  # change port here and in preprocess_parallel.py if 9002 is occupied
        max_char_length=1000000,
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
        process_largefile(
            input_file=in_file,
            output_file=out_file,
            input_file_ids=in_file_index,
            output_index_file=output_index_file,
            function_name=preprocess_parallel.process_document,
            chunk_size=global_options.PARSE_CHUNK_SIZE,
        )
