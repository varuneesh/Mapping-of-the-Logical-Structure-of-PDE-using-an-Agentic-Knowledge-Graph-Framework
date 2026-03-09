import os
from dotenv import load_dotenv

load_dotenv()

MATHPIX_APP_ID = os.getenv("MATHPIX_APP_ID")
MATHPIX_APP_KEY = os.getenv("MATHPIX_APP_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MATHPIX_PDF_URL = "https://api.mathpix.com/v3/pdf"
MATHPIX_STATUS_URL = "https://api.mathpix.com/v3/pdf/{}"