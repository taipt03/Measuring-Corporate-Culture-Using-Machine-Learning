from collections import OrderedDict, Counter
import itertools
from pprint import pprint
import gensim
from pathlib import Path
import global_options
from culture import culture_dictionary

# Load the Word2Vec model
model = gensim.models.Word2Vec.load(
    str(Path(global_options.MODEL_FOLDER, "w2v", "w2v.mod"))
)

# Get the vocabulary size
vocab_number = len(model.wv.key_to_index)
print("Vocab size in the w2v model: {}".format(vocab_number))

# Expand dictionary
expanded_words = culture_dictionary.expand_words_dimension_mean(
    word2vec_model=model,
    seed_words=global_options.SEED_WORDS,
    restrict=global_options.DICT_RESTRICT_VOCAB,
    n=global_options.N_WORDS_DIM,
)
print("Dictionary created. ")

# Ensure one word maps to one dimension only
expanded_words = culture_dictionary.deduplicate_keywords(
    word2vec_model=model,
    expanded_words=expanded_words,
    seed_words=global_options.SEED_WORDS,
)
print("Dictionary deduplicated. ")

# Rank the words under each dimension by similarity to the seed words
expanded_words = culture_dictionary.rank_by_sim(
    expanded_words, global_options.SEED_WORDS, model
)

# Output the dictionary
culture_dictionary.write_dict_to_csv(
    culture_dict=expanded_words,
    file_name=str(Path(global_options.OUTPUT_FOLDER, "dict", "expanded_dict.csv")),
)
print("Dictionary saved at {}".format(str(Path(global_options.OUTPUT_FOLDER, "dict", "expanded_dict.csv"))))
