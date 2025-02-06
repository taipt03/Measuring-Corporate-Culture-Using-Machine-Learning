import datetime
import itertools
import os
import gc
from multiprocessing import Pool
from pathlib import Path
from stanfordnlp.server import CoreNLPClient
import global_options
from culture import file_util, preprocess_parallel

def process_document_with_client(client, doc, doc_id=None):
    """Process a document using an existing CoreNLP client."""
    doc_ann = client.annotate(doc)
    sentences_processed = []
    doc_sent_ids = []
    for i, sentence in enumerate(doc_ann.sentence):
        sentences_processed.append(process_sentence(sentence))
        doc_sent_ids.append(str(doc_id) + "_" + str(i))
    return "\n".join(sentences_processed), "\n".join(doc_sent_ids)

def process_batch(batch_data):
    """Process a batch of documents using a single CoreNLP client."""
    documents, doc_ids = batch_data
    output_lines = []
    output_line_ids = []
    
    # Create a single client for the entire batch
    corenlp_props = {
        "ner.applyFineGrained": "false",
        "annotators": "tokenize, ssplit, pos, lemma, ner, depparse",
        "timeout": "30000",
    }
    
    with CoreNLPClient(
        properties=corenlp_props,
        endpoint="http://localhost:9002",
        start_server=False,
        timeout=120000000
    ) as client:
        for doc, doc_id in zip(documents, doc_ids):
            try:
                output_line, output_line_id = process_document_with_client(client, doc, doc_id)
                output_lines.append(output_line)
                output_line_ids.append(output_line_id)
            except Exception as e:
                print(f"Error processing document {doc_id}: {str(e)}")
                output_lines.append("")
                output_line_ids.append(doc_id)
    
    # Explicitly clear variables
    del client
    gc.collect()
    
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
    chunk_size=100,
    start_index=None,
    batch_size=5
):
    """Process large file with improved memory management."""
    try:
        if start_index is None:
            os.remove(str(output_file))
            os.remove(str(output_index_file))
    except OSError:
        pass

    assert file_util.line_counter(input_file) == len(input_file_ids), \
        "Input file and ID file must have same number of rows."

    # Create a pool with explicit cleanup
    with Pool(global_options.N_CORES) as pool:
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
                line_i += chunk_size
                print(f"{datetime.datetime.now()} - Processing line: {line_i}")

                next_n_lines = list(filter(None.__ne__, next_n_lines))
                next_n_line_ids = list(filter(None.__ne__, next_n_line_ids))

                for i in range(0, len(next_n_lines), batch_size):
                    batch_lines = next_n_lines[i:i + batch_size]
                    batch_ids = next_n_line_ids[i:i + batch_size]
                    
                    results = pool.map(process_batch, [(batch_lines, batch_ids)])
                    
                    for output_lines, output_line_ids in results:
                        write_outputs(output_file, output_index_file, 
                                    output_lines, output_line_ids)
                    
                    # Clear results and force garbage collection
                    del results
                    gc.collect()
                
                # Clear batch data
                del next_n_lines
                del next_n_line_ids
                gc.collect()

def main():
    """Main function to run the processing pipeline."""
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
            chunk_size=global_options.PARSE_CHUNK_SIZE,
            batch_size=5
        )
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()