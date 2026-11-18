from pathlib import Path

FPS = [1.0, 2.0, 5.0, 15.0, 30.0]

PATIENTS = [
    "pat01",
    "pat02",
    "pat03",
    "pat04",
    "pat05",
    "pat06",
    "pat07",
    "pat08",
]

MODELS = [
    "HecVL",
    "SurgVLP",
    "PeskaVLP",
    "SurgCLIP",
    "SurgMotion300M",
    # "SurgMotion1B"
]

PROMPT_STYLES = ["narrative", "descriptive"]

ZERO_SHOT_TEMPERATURE = 0.01

UNLABELLED = -1

PREPROCESSORS = ["native", "letterbox"]

SMOOTHING_SECONDS = [31, 61, 121]

TEMPORAL_METHODS = [
    "none",
    "moving_average",
    "moving_median",
    "exponential_moving_average",
    "hmm_filter",
    "hmm_viterbi",
]

HMM_PSEUDOCOUNT = 1.0

ROOT_DIR = Path(__file__).parent.parent.parent.parent

PHASE_GROUP_FILE = ROOT_DIR / "assets/phases/default.json"

PROMPTS_FILE = ROOT_DIR / "assets/prompts/default.json"

VIDEOS_DIR = ROOT_DIR / "assets/videos"

FRAMES_DIR = ROOT_DIR / "assets/frames"

ANNOTATIONS_DIR = ROOT_DIR / "assets/annotations"

RESULTS_DIR = ROOT_DIR / "results"

EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"

FRAME_CACHE_DIR = Path("/var/tmp/surgphase/frame_cache")

MODEL_WEIGHTS_DIR = ROOT_DIR / "assets/models/weights"

MODEL_NUM_FRAMES = {
    "HecVL": 1,
    "SurgVLP": 1,
    "PeskaVLP": 1,
    "SurgCLIP": 16,
    "SurgMotion300M": 64,
    "SurgMotion1B": 64,
}

TEXT_ENCODER = "emilyalsentzer/Bio_ClinicalBERT"

EMBEDDING_DIM = 768

ENCODING_BATCH_SIZE = 64

CLIP_ENCODING_BATCH_SIZE = 4

__all__ = [
    "FPS",
    "PATIENTS",
    "MODELS",
    "PROMPT_STYLES",
    "ZERO_SHOT_TEMPERATURE",
    "UNLABELLED",
    "PREPROCESSORS",
    "SMOOTHING_SECONDS",
    "TEMPORAL_METHODS",
    "HMM_PSEUDOCOUNT",
    "ROOT_DIR",
    "PHASE_GROUP_FILE",
    "PROMPTS_FILE",
    "VIDEOS_DIR",
    "FRAMES_DIR",
    "ANNOTATIONS_DIR",
    "RESULTS_DIR",
    "EMBEDDINGS_DIR",
    "FRAME_CACHE_DIR",
    "MODEL_WEIGHTS_DIR",
    "MODEL_NUM_FRAMES",
    "TEXT_ENCODER",
    "EMBEDDING_DIM",
    "ENCODING_BATCH_SIZE",
    "CLIP_ENCODING_BATCH_SIZE",
]
