"""Global options for analysis
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

# Hardware options
N_CORES: int = 2  # max number of CPU cores to use
RAM_CORENLP: str = "30G"  # max RAM allocated for parsing using CoreNLP; increase to speed up parsing
PARSE_CHUNK_SIZE: int = 40 # number of lines in the input file to process uing CoreNLP at once. Increase on workstations with larger RAM (e.g. to 1000 if RAM is 64G)  

# Directory locations
os.environ[
    "CORENLP_HOME"
] = "/kaggle/working/Measuring-Corporate-Culture-Using-Machine-Learning/stanford-corenlp-full-2018-10-05"  # location of the CoreNLP models; use / to seperate folders
DATA_FOLDER: str = "data/"
MODEL_FOLDER: str = "models/" # will be created if does not exist
OUTPUT_FOLDER: str = "outputs/" # will be created if does not exist; !!! WARNING: existing files will be removed !!!

# Parsing and analysis options
STOPWORDS: Set[str] = set(
    Path("resources", "StopWords_Generic.txt").read_text().lower().split()
)  # Set of stopwords from https://sraf.nd.edu/textual-analysis/resources/#StopWords
PHRASE_THRESHOLD: int = 10  # threshold of the phraser module (smaller -> more phrases)
PHRASE_MIN_COUNT: int = 10  # min number of times a bigram needs to appear in the corpus to be considered as a phrase
W2V_DIM: int = 300  # dimension of word2vec vectors
W2V_WINDOW: int = 5  # window size in word2vec
W2V_ITER: int = 20  # number of iterations in word2vec
N_WORDS_DIM: int = 500  # max number of words in each dimension of the dictionary
DICT_RESTRICT_VOCAB = None # change to a fraction number (e.g. 0.2) to restrict the dictionary vocab in the top 20% of most frequent vocab

# Inputs for constructing the expanded dictionary
DIMS: List[str] = ["negative", "positive", "risk", "forward-looking", "environmental", "governance", "social", "BC"]
SEED_WORDS: Dict[str, List[str]] = {
     "negative": [
        "able",
        "abundance",
        "acclaimed",
        "accomplish",
        "accomplished",
        "achieve",
        "achievement",
        "attain",
        "attractive",
        "beautiful",
        "beneficial",
        "benefit",
        "best",
        "better",
        "boost",
        "breakthrough",
        "brilliant",
        "charitable",
        "collaborate",
        "compliment",
        "conclusive",
        "confident",
        "constructive",
        "courteous",
        "creative",
        "creativity",
        "delight",
        "dependable",
        "desirable",
        "despite",
        "diligent",
        "distinction",
        "distinctive",
        "dream",
        "easy",
        "effective",
        "efficiency",
        "empower",
        "enable",
        "encourage",
        "enhance",
        "enjoy",
        "enthusiastic",
        "excellence",
        "excellent",
        "excited",
        "exclusive",
        "exceptional",
        "exciting",
        "exemplary",
        "fantastic",
        "favorable",
        "favorite",
        "friendly",
        "gain",
        "good",
        "great",
        "greatest",
        "happiness",
        "happy",
        "honor",
        "ideal",
        "impress",
        "impressive",
        "improve",
        "incredible",
        "innovate",
        "innovation",
        "insightful",
        "inspiration",
        "integrity",
        "invent",
        "invention",
        "leadership",
        "loyal",
        "lucrative",
        "meritorious",
        "optimistic",
        "perfect",
        "pleasant",
        "popular",
        "positive",
        "proactive",
        "proficiency",
        "profitable",
        "progress",
        "prosperous",
        "rebound",
        "receptive",
        "resolve",
        "reward",
        "satisfaction",
        "satisfied",
        "satisfy",
        "smooth",
        "spectacular",
        "stability",
        "stabilize",
        "strength",
        "strong",
        "succeed",
        "success",
        "superior",
        "surpass",
        "transparency",
        "tremendous",
        "unmatched",
        "unparalleled",
        "unsurpassed",
        "upturn",
        "valuable",
        "versatile",
        "vibrant",
        "win",
        "worthy"
    ],
    "positive": [
        "decline",
        "loss",
        "decrease",
        "depreciation",
        "downturn",
        "recession",
        "failure",
        "drop",
        "fall",
        "losses",
        "declining",
        "unprofitable",
        "deficit",
        "underperform",
        "shrink",
        "reduction",
        "falling",
        "weakened",
        "deterioration",
        "insolvency",
        "bankrupt",
        "loss-making",
        "cutback",
        "negative",
        "slump",
        "unfavorable",
        "unsustainable",
        "struggling",
        "deflation"

    ],
    "risk": [
        "loss",
        "decline",
        "decrease",
        "fail",
        "failure",
        "threat",
        "reverse",
        "viable",
        "against",
        "catastrophe",
        "shortage",
        "unable",
        "challenge",
        "uncertain",
        "uncertainty",
        "gain",
        "chance",
        "increase",
        "peak",
        "fluctuate",
        "differ",
        "diversify",
        "probable",
        "probability",
        "significant",
        "climate",
        "floods",
        "carbon",
        "physical",
        "natural",
        "ghg",
        "regulatory",
        "storm",
        "pollution",
        "legal",
        "global",
        "greenhouse",
        "reputation",
        "emissions",
        "change",
        "management",
        "CO2"
    ],
    "forward-looking": [
        "accelerate",
        "anticipate",
        "await",
        "confidence",
        "convince",
        "future",
        "possible",
        "estimate",
        "aim",
        "expect",
        "expectation",
        "forecast",
        "forthcoming",
        "hope",
        "intend",
        "intention",
        "is likely",
        "are likely",
        "is unlikely",
        "are unlikely",
        "look ahead",
        "look forward",
        "next",
        "near term",
        "medium term",
        "optimistic",
        "outlook",
        "plan",
        "predict",
        "prediction",
        "remain",
        "renew",
        "is probable",
        "are probable",
        "probability",
        "opportunity",
        "commitment",
        "further",
        "chance",
        "is well placed",
        "are well placed",
        "is well positioned",
        "are well positioned"
    ],
    "BC": [
        "Blockchain", "block"
    ]
    
}


# Create directories if not exist
Path(DATA_FOLDER, "processed", "parsed").mkdir(parents=True, exist_ok=True)
Path(DATA_FOLDER, "processed", "unigram").mkdir(parents=True, exist_ok=True)
Path(DATA_FOLDER, "processed", "bigram").mkdir(parents=True, exist_ok=True)
Path(DATA_FOLDER, "processed", "trigram").mkdir(parents=True, exist_ok=True)
Path(MODEL_FOLDER, "phrases").mkdir(parents=True, exist_ok=True)
Path(MODEL_FOLDER, "phrases").mkdir(parents=True, exist_ok=True)
Path(MODEL_FOLDER, "w2v").mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER, "dict").mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER, "scores").mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER, "scores", "temp").mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER, "scores", "word_contributions").mkdir(parents=True, exist_ok=True)
