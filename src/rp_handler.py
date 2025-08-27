# at top of rp_handler.py (or speaker_processing.py)
from dotenv import load_dotenv, find_dotenv
import os

# find and load your .env file
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")# 


######SETTING HF_TOKENT#############

import logging
from huggingface_hub import login, whoami
import torch
import numpy as np
from dotenv import load_dotenv, find_dotenv
import base64
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from speechbrain.pretrained import EncoderClassifier # type: ignore

# Removed unused helpers that referenced undefined 'ecapa'
def spk_embed(wave_16k_mono: np.ndarray) -> np.ndarray:
    wav = torch.tensor(wave_16k_mono).unsqueeze(0).to(device)
    return ecapa.encode_batch(wav).squeeze(0).cpu().numpy()

def to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Grab the HF_TOKEN from environment
raw_token = os.environ.get("HF_TOKEN", "")
hf_token = raw_token.strip()

if not hf_token.startswith("hf_"):
    print(f"Token malformed or missing 'hf_' prefix. Forcing correction...")
    hf_token = "h" + hf_token  # Force adding the 'h' (temporary fix)

#print(f" Final HF_TOKEN used: #{hf_token}")
if hf_token:
    try:
        logger.debug(f"HF_TOKEN Loaded: {repr(hf_token[:10])}...")  # Show only start of token for security
        login(token=hf_token, add_to_git_credential=False)  # Safe for container runs
        user = whoami(token=hf_token)
        logger.info(f"Hugging Face Authenticated as: {user['name']}")
    except Exception as e:
        logger.error(" Failed to authenticate with Hugging Face", exc_info=True)
else:
    logger.warning("No Hugging Face token found in HF_TOKEN environment variable.")
##############

from speaker_profiles import load_embeddings, relabel  # top of file
from speaker_processing import process_diarized_output,enroll_profiles, identify_speakers_on_segments, load_known_speakers_from_samples, identify_speaker, relabel_speakers_by_avg_similarity

import shutil
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup
from rp_schema import INPUT_VALIDATIONS
from predict import Predictor, Output
import os
import copy
import logging
import sys
# Create a custom logger
logger = logging.getLogger("rp_handler")
logger.setLevel(logging.DEBUG)  # capture everything at DEBUG or above

# Create console handler and set level to DEBUG
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)

# Create file handler to write logs to 'container_log.txt'
file_handler = logging.FileHandler("container_log.txt", mode="a")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)

# Add both handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)




MODEL = Predictor()
MODEL.setup()

def ensure_job_dir(job_id: str, subdir: str = "input", jobs_directory: str = "/jobs") -> str:
    """Create and return a job subdirectory path."""
    dir_path = os.path.join(jobs_directory, job_id, subdir)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def _strip_data_uri(b64_str: str) -> str:
    """Remove data URI header if present."""
    if b64_str.lstrip().startswith("data:") and "," in b64_str:
        return b64_str.split(",", 1)[1]
    return b64_str

def write_b64_to_file(b64_str: str, dest_path: str) -> int:
    """Append-decoding a single base64 string to file. Returns bytes written."""
    data = base64.b64decode(_strip_data_uri(b64_str), validate=False)
    with open(dest_path, "ab") as f:
        f.write(data)
    return len(data)

def write_b64_chunks_to_file(chunks: list, dest_path: str) -> int:
    """Write list of base64 chunks to file, returns total bytes written."""
    # reset file
    with open(dest_path, "wb") as f:
        pass
    total = 0
    for ch in chunks:
        total += write_b64_to_file(ch, dest_path)
    return total

def materialize_sample_b64(sample: dict, speakers_dir: str, index: int) -> str | None:
    """If sample contains base64, write to a file and set sample['file_path']."""
    ext = sample.get("file_extension") or sample.get("extension") or ".wav"
    if not isinstance(ext, str):
        ext = ".wav"
    if not ext.startswith("."):
        ext = "." + ext
    name = sample.get("name") or f"speaker_{index}"
    safe = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in name)
    dest = os.path.join(speakers_dir, f"{safe}{ext}")
    if sample.get("b64_chunks"):
        write_b64_chunks_to_file(sample["b64_chunks"], dest)
        sample["file_path"] = dest
        sample.setdefault("name", name)
        return dest
    if sample.get("b64"):
        write_b64_chunks_to_file([sample["b64"]], dest)
        sample["file_path"] = dest
        sample.setdefault("name", name)
        return dest
    return None


def cleanup_job_files(job_id, jobs_directory='/jobs'):
    job_path = os.path.join(jobs_directory, job_id)
    if os.path.exists(job_path):
        try:
            shutil.rmtree(job_path)
            logger.info(f"Removed job directory: {job_path}")
        except Exception as e:
            logger.error(f"Error removing job directory {job_path}: {str(e)}", exc_info=True)
    else:
        logger.debug(f"Job directory not found: {job_path}")

# --------------------------------------------------------------------
# main serverless entry-point
# --------------------------------------------------------------------
error_log = []
def run(job):
    job_id     = job["id"]
    job_input  = job["input"]

    # ------------- validate basic schema ----------------------------
    validated = validate(job_input, INPUT_VALIDATIONS)
    if "errors" in validated:
        return {"error": validated["errors"]}

    # ------------- 1) load/assemble primary audio -------------------
    audio_file_path = None
    try:
        if job_input.get("audio_file"):
            audio_file_path = download_files_from_urls(job_id, [job_input["audio_file"]])[0]
            logger.debug(f"Audio downloaded → {audio_file_path}")
        elif job_input.get("audio_b64_chunks") or job_input.get("audio_b64"):
            input_dir = ensure_job_dir(job_id, "input")
            ext = job_input.get("audio_extension", ".wav") or ".wav"
            if not isinstance(ext, str):
                ext = ".wav"
            if not ext.startswith("."):
                ext = "." + ext
            audio_file_path = os.path.join(input_dir, f"audio{ext}")
            if job_input.get("audio_b64_chunks"):
                total = write_b64_chunks_to_file(job_input["audio_b64_chunks"], audio_file_path)
                logger.info(f"Audio assembled from {len(job_input['audio_b64_chunks'])} base64 chunks → {audio_file_path} ({total} bytes)")
            else:
                total = write_b64_chunks_to_file([job_input["audio_b64"]], audio_file_path)
                logger.info(f"Audio assembled from single base64 → {audio_file_path} ({total} bytes)")
        else:
            return {"error": "No audio provided. Supply 'audio_file' URL or 'audio_b64'/'audio_b64_chunks'."}
    except Exception as e:
        logger.error("Audio input preparation failed", exc_info=True)
        return {"error": f"audio input: {e}"}

    # ------------- 2) speaker profiles (optional, allow base64) -----
    speaker_profiles = job_input.get("speaker_samples", [])
    embeddings = {}
    if speaker_profiles:
        # materialize any base64 speaker samples to files
        try:
            speakers_dir = ensure_job_dir(job_id, "speakers")
            for idx, sample in enumerate(speaker_profiles):
                if isinstance(sample, dict) and (sample.get("b64_chunks") or sample.get("b64")):
                    path = materialize_sample_b64(sample, speakers_dir, idx)
                    if path:
                        logger.debug(f"Speaker sample materialized → {path}")
        except Exception as e:
            logger.error("Failed to materialize speaker samples", exc_info=True)
            error_log.append(f"speaker sample materialization: {e}")
        try:
            embeddings = load_known_speakers_from_samples(
                speaker_profiles,
                huggingface_access_token=hf_token  # or job_input.get("huggingface_access_token")
            )
            logger.info(f"Enrolled {len(embeddings)} speaker profiles successfully.")
        except Exception as e:
            logger.error("Enrollment failed", exc_info=True)
            error_log.append(f"Enrollment skipped: {e}")
    # ----------------------------------------------------------------

    # ------------- 3) call WhisperX / VAD / diarization -------------
    predict_input = {
        "audio_file"               : audio_file_path,
        "language"                 : job_input.get("language"),
        "language_detection_min_prob": job_input.get("language_detection_min_prob", 0),
        "language_detection_max_tries": job_input.get("language_detection_max_tries", 5),
        "initial_prompt"           : job_input.get("initial_prompt"),
        "batch_size"               : job_input.get("batch_size", 64),
        "temperature"              : job_input.get("temperature", 0),
        "vad_onset"                : job_input.get("vad_onset", 0.50),
        "vad_offset"               : job_input.get("vad_offset", 0.363),
        "align_output"             : job_input.get("align_output", False),
        "diarization"              : job_input.get("diarization", False),
        "huggingface_access_token" : job_input.get("huggingface_access_token"),
        "min_speakers"             : job_input.get("min_speakers"),
        "max_speakers"             : job_input.get("max_speakers"),
        "debug"                    : job_input.get("debug", False),
    }

    try:
        result = MODEL.predict(**predict_input)             # <-- heavy job
    except Exception as e:
        logger.error("WhisperX prediction failed", exc_info=True)
        return {"error": f"prediction: {e}"}

    output_dict = {
        "segments"         : result.segments,
        "detected_language": result.detected_language
    }
    # ------------------------------------------------embedding-info----------------
    # 4) speaker verification (optional)
    if embeddings:
        try:
            segments_with_speakers = identify_speakers_on_segments(
                segments=output_dict["segments"],
                audio_path=audio_file_path,
                enrolled=embeddings,
                threshold=0.1  # Adjust threshold as needed
            )
            segments_with_final_labels = relabel_speakers_by_avg_similarity(segments_with_speakers)
            output_dict["segments"] = segments_with_final_labels
            logger.info("Speaker identification completed successfully.")
        except Exception as e:
            logger.error("Speaker identification failed", exc_info=True)
            output_dict["warning"] = f"Speaker identification skipped: {e}"
    else:
        logger.info("No enrolled embeddings available; skipping speaker identification.")

    # 4-Cleanup and return output_dict normally
    try:
        rp_cleanup.clean(["input_objects"])
        cleanup_job_files(job_id)
    except Exception as e:
        logger.warning(f"Cleanup issue: {e}", exc_info=True)

    if error_log:
        output_dict["warnings"] = error_log

    return output_dict

runpod.serverless.start({"handler": run})


#     embeddings = {} # ensure the name is always bound
#     if job_input.get("speaker_verification", True):
#         logger.info(f"Speaker-verification requested: True")
#         try:
#             embeddings = load_known_speakers_from_samples(
#                 speaker_profiles,
#                 huggingface_access_token=predict_input["huggingface_access_token"]
#             )
#             logger.info(f"  • Enrolled {len(embeddings)} profiles")
#         except Exception as e:
#             logger.error("Failed loading speaker profiles", exc_info=True)
#             output_dict["warning"] = f"enrollment skipped: {e}"

#         embedding_log_data = None  # Initialize here to avoid UnboundLocalError

#         if embeddings:  # only attempt verification if we actually got something
#             try:
#                 output_dict, embedding_log_data = process_diarized_output(
#                     output_dict,
#                     audio_file_path,
#                     embeddings,
#                     huggingface_access_token=job_input.get("huggingface_access_token"),
#                     return_logs=False # <-- set to True for debugging
#             except Exception as e:
#                 logger.error("Error during speaker verification", exc_info=True)
#                 output_dict["warning"] = f"verification skipped: {e}"
#         else:
#             logger.info("No embeddings to verify against; skipping verification step")

#     if embedding_log_data:
#         output_dict["embedding_logs"] = embedding_log_data

#     # 5) cleanup
#     try:
#         rp_cleanup.clean(["input_objects"])
#         cleanup_job_files(job_id)
#     except Exception as e:
#         logger.warning(f"Cleanup issue: {e}")

#         # If you have any errors, attach them to the output
#     if error_log:
#         output_dict["error_log"] = "\n".join(error_log)

#     return output_dict

# runpod.serverless.start({"handler": run})
