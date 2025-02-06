"""Global options for analysis
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

# Hardware options
N_CORES: int = 4  # max number of CPU cores to use
RAM_CORENLP: str = "30G"  # max RAM allocated for parsing using CoreNLP; increase to speed up parsing
PARSE_CHUNK_SIZE: int = 5 # number of lines in the input file to process uing CoreNLP at once. Increase on workstations with larger RAM (e.g. to 1000 if RAM is 64G)  

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
    "environmental": [
        "clean", "environmental", "epa", "sustainability", "climate", "warming", "biofuel", "biofuels", "green", "renewable", 
        "solar", "stewardship", "wind", "atmosphere", "emission", "emissions", "emit", "ghg", "ghgs", "greenhouse", 
        "agriculture", "deforestation", "pesticide", "pesticides", "wetlands", "zoning", "biodiversity", "species", 
        "wilderness", "wildlife", "freshwater", "groundwater", "water", "cleaner", "cleanup", "coal", "contamination", 
        "fossil", "resource", "air", "carbon", "nitrogen", "pollution", "superfund", "biphenyls", "hazardous", "householding", 
        "pollutants", "printing", "recycle", "recycling", "toxic", "waste", "wastes", "weee"
    ],
    "governance": [
        "align", "aligned", "aligning", "alignment", "aligns", "bylaw", "bylaws", "charter", "charters", "culture", 
        "death", "duly", "independent", "parents", "cobc", "ethic", "ethical", "ethically", "ethics", "honesty", "bribery", 
        "corrupt", "corruption", "crimes", "embezzlement", "grassroots", "influence", "influences", "influencing", "lobbied", 
        "lobbies", "lobby", "lobbying", "lobbyist", "lobbyists", "whistleblower", "compliance", "conduct", "conformity", 
        "governance", "misconduct", "parachute", "parachutes", "perquisites", "plane", "planes", "poison", "retirement", 
        "approval", "approvals", "approve", "approved", "approves", "approving", "assess", "assessed", "assesses", 
        "assessing", "assessment", "assessments", "audit", "audited", "auditing", "auditor", "auditors", "audits", "control", 
        "controls", "coso", "detect", "detected", "detecting", "detection", "evaluate", "evaluated", "evaluates", "evaluating", 
        "evaluation", "evaluations", "examination", "examinations", "examine", "examined", "examines", "examining", "irs", 
        "oversee", "overseeing", "oversees", "oversight", "review", "reviewed", "reviewing", "reviews", "rotation", "test", 
        "tested", "testing", "tests", "treadway", "backgrounds", "independence", "leadership", "nomination", "nominations", 
        "nominee", "nominees", "perspectives", "qualifications", "refreshment", "skill", "skills", "succession", "tenure", 
        "vacancies", "vacancy", "appreciation", "award", "awarded", "awarding", "awards", "bonus", "bonuses", "cd", "compensate", 
        "compensated", "compensates", "compensating", "compensation", "eip", "iso", "isos", "payout", "payouts", "pension", 
        "prsu", "prsus", "recoupment", "remuneration", "reward", "rewarding", "rewards", "rsu", "rsus", "salaries", "salary", 
        "severance", "vest", "vested", "vesting", "vests", "ballot", "ballots", "cast", "consent", "elect", "elected", "electing", 
        "election", "elections", "elects", "nominate", "nominated", "plurality", "proponent", "proponents", "proposal", 
        "proposals", "proxies", "quorum", "vote", "voted", "votes", "voting", "attract", "attracting", "attracts", "incentive", 
        "incentives", "interview", "interviews", "motivate", "motivated", "motivates", "motivating", "motivation", "recruit", 
        "recruiting", "recruitment", "retain", "retainer", "retainers", "retaining", "retention", "talent", "talented", "talents", 
        "brother", "clicking", "conflict", "conflicts", "family", "grandchildren", "grandparent", "grandparents", "inform", 
        "insider", "insiders", "inspector", "inspectors", "interlocks", "nephews", "nieces", "posting", "relatives", "siblings", 
        "sister", "son", "spousal", "spouse", "spouses", "stepchildren", "stepparents", "transparency", "transparent", "visit", 
        "visiting", "visits", "webpage", "website", "announce", "announced", "announcement", "announcements", "announces", 
        "announcing", "communicate", "communicated", "communicates", "communicating", "erm", "fairly", "integrity", "liaison", 
        "presentation", "presentations", "sustainable", "asc", "disclose", "disclosed", "discloses", "disclosing", "disclosure", 
        "disclosures", "fasb", "gaap", "objectivity", "press", "sarbanes", "engagement", "engagements", "feedback", "hotline", 
        "investor", "invite", "invited", "mail", "mailed", "mailing", "mailings", "notice", "relations", "stakeholder", "stakeholders", 
        "compact"
    ],
    "social": [
        "citizen", "citizens", "csr", "disabilities", "disability", "disabled", "human", "nations", "social", "un", "veteran", 
        "veterans", "vulnerable", "dignity", "discriminate", "discriminated", "discriminating", "discrimination", "equality", 
        "freedom", "humanity", "nondiscrimination", "sexual", "communities", "community", "expression", "marriage", "privacy", 
        "peace", "bargaining", "eeo", "fairness", "fla", "harassment", "injury", "labor", "overtime", "ruggie", "sick", "wage", 
        "wages", "workplace", "bisexual", "diversity", "ethnic", "ethnically", "ethnicities", "ethnicity", "female", "females", 
        "gay", "gays", "gender", "genders", "homosexual", "immigration", "lesbian", "lesbians", "lgbt", "minorities", "minority", 
        "ms", "race", "racial", "religion", "religious", "sex", "transgender", "woman", "women", "occupational", "safe", "safely", 
        "safety", "ilo", "labour", "eicc", "children", "epidemic", "health", "healthy", "ill", "illness", "pandemic", "childbirth", 
        "drug", "medicaid", "medicare", "medicine", "medicines", "hiv", "alcohol", "drinking", "bugs", "conformance", "defects", 
        "fda", "inspection", "inspections", "minerals", "standardization", "warranty", "endowment", "endowments", "people", 
        "philanthropic", "philanthropy", "socially", "societal", "society", "welfare", "charitable", "charities", "charity", 
        "donate", "donated", "donates", "donating", "donation", "donations", "donors", "foundation", "foundations", "gift", "gifts", 
        "nonprofit", "poverty", "courses", "educate", "educated", "educates", "educating", "education", "educational", "learning", 
        "mentoring", "scholarships", "teach", "teacher", "teachers", "teaching", "training", "employ", "employment", "headcount", 
        "hire", "hired", "hires", "hiring", "staffing", "unemployment"
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
