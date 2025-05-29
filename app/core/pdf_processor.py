import subprocess
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_pdf(pdf_path: str) -> str | None:
    """Extracts text content from a PDF file using pdftotext.

    Args:
        pdf_path: The absolute path to the PDF file.

    Returns:
        The extracted text content as a string, or None if extraction fails.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found: {pdf_path}")
        return None

    try:
        # Use pdftotext to extract text to stdout (-)
        # The arguments ensure layout preservation (-layout) and UTF-8 encoding (-enc UTF-8)
        # Using '-' sends output to stdout
        result = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", pdf_path, "-"],
            capture_output=True,
            text=True,
            check=True,  # Raise CalledProcessError on failure
            encoding='utf-8' # Specify encoding for text=True
        )
        logging.info(f"Successfully extracted text from {pdf_path}")
        return result.stdout
    except FileNotFoundError:
        logging.error("pdftotext command not found. Ensure poppler-utils is installed.")
        return None
    except subprocess.CalledProcessError as e:
        logging.error(f"pdftotext failed for {pdf_path}: {e}")
        logging.error(f"pdftotext stderr: {e.stderr}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during PDF text extraction for {pdf_path}: {e}")
        return None

# Example usage (for testing purposes):
# if __name__ == '__main__':
#     # Create a dummy PDF for testing if needed, or use an existing one
#     # For now, just define the function
#     pass

