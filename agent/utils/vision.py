"""
👁️ MediBlaze Vision Module
Analyzes medical images (symptoms, skin conditions, prescriptions, X-rays, etc.)
using Google Gemini's multimodal capabilities.

The result is then passed to the main Groq agent for deeper medical context + RAG retrieval.
"""

import os
import base64
import logging
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("❌ GOOGLE_API_KEY is missing. Add it to your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)


def analyze_medical_image(image_bytes: bytes, mime_type: str, user_question: str = "") -> str:
    """
    Analyze a medical image using Gemini Vision.

    Args:
        image_bytes: Raw bytes of the uploaded image.
        mime_type:   MIME type of the image (e.g. "image/jpeg", "image/png").
        user_question: Optional question the user asked about the image.

    Returns:
        A structured medical analysis as a string.
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-lite")

        prompt = (
            "You are MediBlaze 🏥, an advanced AI medical assistant with expert-level "
            "medical image analysis capabilities.\n\n"
            "Analyze the uploaded medical image and provide a comprehensive structured response.\n\n"
            "Your analysis must cover (as applicable to what you see):\n"
            "1. **🔍 Image Type & Content**: What type of image is this? "
            "(e.g. skin condition, prescription, X-ray, medical report, eye, wound, rash, etc.)\n"
            "2. **👁️ Visual Observations**: Describe exactly what you observe in detail.\n"
            "3. **🩺 Medical Interpretation**: What could this indicate medically? "
            "List possible conditions or diagnoses based on what you see.\n"
            "4. **💊 If it's a prescription**: Extract and list all medications, dosages, "
            "frequency, and any special instructions clearly.\n"
            "5. **⚠️ Warning Signs**: Any urgent or concerning findings that need immediate attention?\n"
            "6. **✅ Recommended Next Steps**: What should the patient do next?\n\n"
            f"{'**Patient question:** ' + user_question if user_question else ''}\n\n"
            "> ⚠️ **Important Disclaimer**: This analysis is for educational purposes only. "
            "Always consult a qualified healthcare professional for diagnosis and treatment."
        )

        image_part = {
            "mime_type": mime_type,
            "data": base64.b64encode(image_bytes).decode("utf-8"),
        }

        logger.info("👁️ [MediBlaze Vision] Sending image to Gemini for analysis...")
        response = model.generate_content([prompt, image_part])
        logger.info("✅ [MediBlaze Vision] Image analysis complete.")
        return response.text

    except Exception as e:
        logger.error(f"❌ [MediBlaze Vision] Error analyzing image: {str(e)}")
        raise RuntimeError(f"Image analysis failed: {str(e)}")
