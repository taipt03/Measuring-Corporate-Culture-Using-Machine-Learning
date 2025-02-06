import datetime
import itertools
import os
from multiprocessing import Pool, Manager
from pathlib import Path

from stanfordnlp.server import CoreNLPClient

import global_options
from culture import file_util, preprocess_parallel

# Initialize a global manager dictionary to store the shared CoreNLPClient
manager = Manager()
shared_client = manager.dict()

def init_client():
    """Initialize CoreNLPClient once per process"""
    global shared_client
    if "client" not in shared_client:
        shared_client["client"] = CoreNLPClient(
            endpoint="http://localhost:9002",
            start_server=False,
            timeout=120000000,
        )

def process_document(doc, doc_id=None):
    """Process a document using the shared CoreNLPClient."""
    client = shared_client["client"]
    doc_ann = client.annotate(doc)
    sentences_processed = []
    doc_sent_ids = []
    for i, sentence in enumerate(doc_ann.sentence):
        sentences_processed.append(process_sentence(sentence))
        doc_sent_ids.append(str(doc_id) + "_" + str(i))
    return "\n".join(sentences_processed), "\n".join(doc_sent_ids)

def process_largefile(
    input_file,
    output_file,
    input_file_ids,
    output_index_file,
    function_name,
    chunk_size=100,
    start_index=None,
):
    """Processes a large file in chunks with multiprocessing."""
    try:
        if start_index is None:
            os.remove(str(output_file))
            os.remove(str(output_index_file))
    except OSError:
        pass
    
    assert file_util.line_counter(input_file) == len(input_file_ids), "Mismatched input file and ID file length."
    
    with open(input_file, newline="\n", encoding="utf-8", errors="ignore") as f_in:
        line_i = 0
        if start_index is not None:
            for _ in range(start_index):
                next(f_in)
            input_file_ids = input_file_ids[start_index:]
            line_i = start_index
        
        with Pool(global_options.N_CORES, initializer=init_client) as pool:
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
                
                for output_line, output_line_id in pool.starmap(
                    function_name, zip(next_n_lines, next_n_line_ids)
                ):
                    output_lines.append(output_line)
                    output_line_ids.append(output_line_id)
                
                with open(output_file, "a", newline="\n") as f_out:
                    f_out.write("\n".join(output_lines) + "\n")
                if output_index_file is not None:
                    with open(output_index_file, "a", newline="\n") as f_out:
                        f_out.write("\n".join(output_line_ids) + "\n")

if __name__ == "__main__":
    with CoreNLPClient(
        properties={
            "ner.applyFineGrained": "false",
            "annotators": "tokenize, ssplit, pos, lemma, ner, depparse",
        },
        memory=global_options.RAM_CORENLP,
        threads=global_options.N_CORES,
        timeout=12000000,
        endpoint="http://localhost:9002",
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
            function_name=process_document,
            chunk_size=global_options.PARSE_CHUNK_SIZE,
        )
