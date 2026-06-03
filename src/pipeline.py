import logging

from src.config import INPUT_PATH, MODEL_PATH, OUTPUT_PATH
from src.load_data import load_input_data
from src.preprocess import preprocess_data
from src.save_submission import save_submission
from src.score import load_model, score_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(input_path=INPUT_PATH, model_path=MODEL_PATH, output_path=OUTPUT_PATH):
    """Run the full CPU-only batch inference pipeline."""
    logger.info("Starting fraud inference pipeline")
    raw_data = load_input_data(input_path)
    features = preprocess_data(raw_data)
    model = load_model(model_path)
    predictions = score_data(model, features)
    submission = save_submission(predictions, output_path)
    logger.info("Pipeline finished successfully: %s", output_path)
    return submission


if __name__ == "__main__":
    run_pipeline()
