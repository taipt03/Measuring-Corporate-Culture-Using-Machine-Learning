from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

def filter(text: str):
    doc = nlp(text)
    
    masked_text = text
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
            # Mask the entity
            masked_text = masked_text.replace(ent.text, "")
            
    res = []
    sentences = masked_text.split('.')
    
    for sentence in sentences:
        t_sentence = []
        words = sentence.split(' ')
        
        for word in words:
            if word.strip().lower() in ENGLISH_STOP_WORDS or word.strip().isdigit():
                continue
            
            t_sentence.append(word.strip())
            
        res.append(' '.join(t_sentence))
        
    return '.'.join(res)

def process_file(input_file: str, output_file: str = None):
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    
    filtered_text = filter(text)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(filtered_text)
        print(f"Filtered text saved to {output_file}")
    else:
        print(filtered_text)

process_file("C:/Users/tuant/Downloads/OneDrive_2024-12-08 (2)/extracted/266_ANNUAL_2013_ENG.txt", "C:/Users/tuant/Downloads/OneDrive_2024-12-08 (2)/extracted/266_ANNUAL_2013_ENG_OUTPUT.txt")
