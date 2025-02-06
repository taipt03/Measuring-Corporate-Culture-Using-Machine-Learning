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
    # ... existing code ...
    
    for next_n_lines, next_n_line_ids in zip(
        itertools.zip_longest(*[f_in] * chunk_size),
        itertools.zip_longest(*[iter(input_file_ids)] * chunk_size),
    ):
        line_i += chunk_size
        print(datetime.datetime.now())
        print(f"Processing line: {line_i}.")
        next_n_lines = list(filter(None.__ne__, next_n_lines))
        next_n_line_ids = list(filter(None.__ne__, next_n_line_ids))
        output_lines = []
        output_line_ids = []
        
        # Create and close pool for each chunk to force cleanup
        with Pool(global_options.N_CORES) as pool:
            results = pool.starmap(
                function_name, zip(next_n_lines, next_n_line_ids)
            )
            # Process results immediately
            for output_line, output_line_id in results:
                output_lines.append(output_line)
                output_line_ids.append(output_line_id)
            
        # Write and clear immediately
        output_text = "\n".join(output_lines) + "\n"
        output_ids = "\n".join(output_line_ids) + "\n"
        
        with open(output_file, "a", newline="\n") as f_out:
            f_out.write(output_text)
        if output_index_file is not None:
            with open(output_index_file, "a", newline="\n") as f_out:
                f_out.write(output_ids)
                
        # Force cleanup
        del output_lines
        del output_line_ids
        del output_text
        del output_ids


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
