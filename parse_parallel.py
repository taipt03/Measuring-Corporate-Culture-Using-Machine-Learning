from pathlib import Path
from stanfordnlp.server import CoreNLPClient
from multiprocessing import Pool
import itertools
import datetime
import os
import gc

class DocumentProcessor:
    """A class to handle document processing with proper resource management"""
    def __init__(self, port=9002):
        self.client = CoreNLPClient(
            endpoint=f"http://localhost:{port}",
            start_server=False,
            timeout=120000000
        )
        self.client.start()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.stop()
        gc.collect()
        
    def process_document(self, doc, doc_id=None):
        """Process a single document using a persistent CoreNLP client
        
        Arguments:
            doc {str} -- raw string of a document
            doc_id {str} -- raw string of a document ID
        
        Returns:
            tuple -- (processed_sentences, sentence_ids)
        """
        if not doc:
            return "", ""
            
        try:
            doc_ann = self.client.annotate(doc)
            sentences_processed = []
            doc_sent_ids = []
            
            for i, sentence in enumerate(doc_ann.sentence):
                sentences_processed.append(process_sentence(sentence))
                doc_sent_ids.append(f"{doc_id}_{i}")
                
            return "\n".join(sentences_processed), "\n".join(doc_sent_ids)
            
        except Exception as e:
            print(f"Error processing document {doc_id}: {str(e)}")
            return "", ""

def process_document_batch(batch_data):
    """Process a batch of documents using a single CoreNLP client instance"""
    docs, doc_ids = batch_data
    processed_docs = []
    processed_ids = []
    
    # Create a single client for the entire batch
    with DocumentProcessor() as processor:
        for doc, doc_id in zip(docs, doc_ids):
            processed_text, processed_id = processor.process_document(doc, doc_id)
            processed_docs.append(processed_text)
            processed_ids.append(processed_id)
    
    return "\n".join(processed_docs), "\n".join(processed_ids)

def process_largefile(
    input_file,
    output_file,
    input_file_ids,
    output_index_file,
    chunk_size=100,
    start_index=None,
):
    """Memory-optimized version of the large file processor"""
    try:
        if start_index is None:
            os.remove(str(output_file))
            os.remove(str(output_index_file))
    except OSError:
        pass
    
    assert file_util.line_counter(input_file) == len(input_file_ids), \
        "Input file and ID file must have same number of rows"

    with open(input_file, newline="\n", encoding="utf-8", errors="ignore") as f_in:
        line_i = 0
        
        if start_index is not None:
            for _ in range(start_index):
                next(f_in)
            input_file_ids = input_file_ids[start_index:]
            line_i = start_index

        # Create a single pool for all processing
        with Pool(global_options.N_CORES) as pool:
            for next_n_lines, next_n_line_ids in zip(
                itertools.zip_longest(*[f_in] * chunk_size),
                itertools.zip_longest(*[iter(input_file_ids)] * chunk_size),
            ):
                line_i += chunk_size
                print(f"{datetime.datetime.now()} - Processing line: {line_i}")
                
                # Filter None values
                next_n_lines = list(filter(None.__ne__, next_n_lines))
                next_n_line_ids = list(filter(None.__ne__, next_n_line_ids))
                
                # Split data into smaller batches for better memory management
                batch_size = max(chunk_size // global_options.N_CORES, 10)
                batches = [
                    (next_n_lines[i:i + batch_size], 
                     next_n_line_ids[i:i + batch_size])
                    for i in range(0, len(next_n_lines), batch_size)
                ]
                
                # Process batches
                results = pool.map(process_document_batch, batches)
                
                # Write results immediately
                for processed_docs, processed_ids in results:
                    if processed_docs:
                        with open(output_file, "a", newline="\n") as f_out:
                            f_out.write(processed_docs + "\n")
                    if processed_ids and output_index_file:
                        with open(output_index_file, "a", newline="\n") as f_out:
                            f_out.write(processed_ids + "\n")
                
                # Force garbage collection after each chunk
                gc.collect()

if __name__ == "__main__":
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
        chunk_size=global_options.PARSE_CHUNK_SIZE,
    )