import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\')

# NLLB-200 Language Code Mappings (Add more as needed based on the 10+ requirement)
# Using Flores-200 codes: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
NLLB_LANG_MAP = {
    "english": "eng_Latn",
    "hindi": "hin_Deva",
    "spanish": "spa_Latn",
    "french": "fra_Latn",
    "german": "deu_Latn",
    "arabic": "arb_Arab",
    "chinese": "zho_Hans", # Simplified Chinese
    "japanese": "jpn_Jpan",
    "tamil": "tam_Taml",
    "bengali": "ben_Beng",
    # Add other major languages if feasible within NLLB-200 scope
}

DEFAULT_MODEL_NAME = "facebook/nllb-200-distilled-600M" # Using distilled version for efficiency

class TranslationService:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.translator = None
        self._load_model()

    def _load_model(self):
        """Loads the NLLB translation model and tokenizer."""
        try:
            logging.info(f"Loading NLLB model: {self.model_name}")
            # Check if CUDA is available and set device
            device = 0 if torch.cuda.is_available() else -1
            if device == 0:
                logging.info("CUDA detected. Loading model on GPU.")
            else:
                logging.info("CUDA not detected. Loading model on CPU.")

            # Load model and tokenizer
            # Using AutoModel/AutoTokenizer handles potential configuration needs
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

            # Create the pipeline
            # Specify device explicitly. device=0 for GPU 0, device=-1 for CPU.
            self.translator = pipeline(
                "translation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device
            )
            logging.info("NLLB translation model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load NLLB model {self.model_name}: {e}")
            # Consider fallback mechanisms or raising the error
            self.translator = None

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str | None:
        """Translates text from source language to target language.

        Args:
            text: The text to translate.
            src_lang: The source language (e.g., 'english', 'hindi').
            tgt_lang: The target language (e.g., 'french', 'german').

        Returns:
            The translated text, or None if translation fails.
        """
        if not self.translator:
            logging.error("Translation model is not loaded. Cannot translate.")
            return None

        src_code = NLLB_LANG_MAP.get(src_lang.lower())
        tgt_code = NLLB_LANG_MAP.get(tgt_lang.lower())

        if not src_code:
            logging.warning(f"Source language 	'{src_lang}	' not supported by NLLB mapping.")
            return None # Or handle fallback
        if not tgt_code:
            logging.warning(f"Target language 	'{tgt_lang}	' not supported by NLLB mapping.")
            return None # Or handle fallback

        try:
            logging.info(f"Translating from {src_lang} ({src_code}) to {tgt_lang} ({tgt_code})")
            # NLLB pipeline requires src_lang and tgt_lang parameters with the specific codes
            result = self.translator(text, src_lang=src_code, tgt_lang=tgt_code)
            translated_text = result[0]["translation_text"]
            logging.info("Translation successful.")
            return translated_text
        except Exception as e:
            logging.error(f"Error during translation from {src_lang} to {tgt_lang}: {e}")
            return None

# Example usage (for testing):
# if __name__ == '__main__':
#     # Ensure model is downloaded first (might take time and disk space)
#     # Set TRANSFORMERS_CACHE environment variable if needed
#     translator_service = TranslationService()
#     if translator_service.translator:
#         english_text = "Hello, how are you?"
#         hindi_translation = translator_service.translate(english_text, "english", "hindi")
#         print(f"English: {english_text}")
#         print(f"Hindi: {hindi_translation}")
#
#         french_text = "Bonjour le monde!"
#         english_translation = translator_service.translate(french_text, "french", "english")
#         print(f"French: {french_text}")
#         print(f"English: {english_translation}")

