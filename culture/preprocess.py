import os
import re
import functools
import spacy
from transformers import pipeline

class Preprocessor:
    def __init__(self):
        # Load the NER model using Hugging Face Transformers
        self.ner_pipeline = pipeline("ner", device=0)  # Use GPU
        self.nlp = spacy.load("en_core_web_sm")  # Load spaCy model for lemmatization

    def process_document(self, doc, doc_id=None):
        """Main method: Annotate a document using Hugging Face NER model"""
        sentences = doc.split('. ')
        sentences_processed = []
        doc_ids = []
        for i, sentence in enumerate(sentences):
            processed_sentence = self.process_sentence(sentence)
            sentences_processed.append(processed_sentence)
            doc_ids.append(f"{doc_id}_{i}")
        return sentences_processed, doc_ids

    def process_sentence(self, sentence):
        """Process a raw sentence"""
        # Perform NER
        ner_results = self.ner_pipeline(sentence)

        # Lemmatization
        doc = self.nlp(sentence)
        lemmatized_sentence = " ".join([token.lemma_ for token in doc])

        # Tagging NER
        for entity in ner_results:
            start = entity['start']
            end = entity['end']
            label = entity['entity']
            lemmatized_sentence = lemmatized_sentence[:start] + f"[NER:{label}]" + lemmatized_sentence[start:end] + lemmatized_sentence[end:]
        
        # Handle MWEs (basic example)
        lemmatized_sentence = self.handle_mwes(lemmatized_sentence)

        return lemmatized_sentence

    def handle_mwes(self, sentence):
        """Concatenate MWEs (example)"""
        # Define MWEs (this is just an example; you can define your own list)
        mwes = {
            "Stanford University": "Stanford_University",
            "New York City": "New_York_City",
        }
        for mwe, replacement in mwes.items():
            sentence = sentence.replace(mwe, replacement)
        return sentence

class TextCleaner:
    """Clean the text parsed by the preprocessor"""

    def __init__(self):
        pass

    def remove_NER(self, line):
        """Remove the named entity and only leave the tag"""
        NERs = re.compile(r"(\[NER:\w+\])(\S+)")
        line = re.sub(NERs, r"\1", line)
        return line

    def remove_punct_num(self, line):
        """Remove tokens that are only numerics and punctuation marks"""
        tokens = line.strip().lower().split(" ")
        tokens = [re.sub(r"\[pos:.*?\]", "", t) for t in tokens]
        puncts_stops = set(["-lrb-", "-rrb-", "-lsb-", "-rsb-", "'s"])
        tokens = [
            t for t in tokens if any(c.isalpha() for c in t) and t not in puncts_stops and len(t) > 1
        ]
        return " ".join(tokens)

    def clean(self, line, id):
        """Main function that chains all filters together and applies to a string."""
        return (
            functools.reduce(
                lambda obj, func: func(obj),
                [self.remove_NER, self.remove_punct_num],
                line,
            ),
            id,
        )

if __name__ == "__main__":
    # Example usage
    doc = "When I was a child in Ohio, I always wanted to go to Stanford University. But I went along with my parents."
    EC_preprocessor = Preprocessor()
    sentences_processed, doc_ids = EC_preprocessor.process_document(doc, "doc1")

    for sentence, doc_id in zip(sentences_processed, doc_ids):
        cleaned_sentence = TextCleaner().clean(sentence, doc_id)
        print(f"Processed Sentence ID {doc_id}: {cleaned_sentence}")