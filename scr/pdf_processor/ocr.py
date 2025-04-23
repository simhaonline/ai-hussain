"""
OCR module for processing scanned PDF pages.
"""
import logging
import pytesseract
from PIL import Image
import numpy as np
from typing import Optional, Union

from config.config import settings

logger = logging.getLogger(__name__)

def perform_ocr(image: Union[Image.Image, np.ndarray, str], 
                language: str = settings.OCR_LANGUAGE,
                config: Optional[str] = None) -> str:
    """
    Perform OCR on an image.
    
    Args:
        image: PIL Image, numpy array, or path to image file
        language: OCR language
        config: Tesseract configuration string
        
    Returns:
        Extracted text
    """
    try:
        logger.debug(f"Performing OCR with language: {language}")
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Use pytesseract to extract text
        ocr_config = config or "--psm 1 --oem 3"
        text = pytesseract.image_to_string(image, lang=language, config=ocr_config)
        
        # Clean up the text
        text = text.strip()
        
        logger.debug(f"OCR extracted {len(text)} characters")
        return text
        
    except Exception as e:
        logger.error(f"OCR processing error: {e}")
        return ""

def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Preprocess an image to enhance OCR results.
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale if it's not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Increase contrast
    import PIL.ImageOps
    image = PIL.ImageOps.autocontrast(image)
    
    # Optional: Apply thresholding for binary image
    # This can help with text extraction in some cases
    from PIL import ImageFilter
    image = image.filter(ImageFilter.SHARPEN)
    
    # Resize if too small
    min_dpi = 300
    if image.width < min_dpi or image.height < min_dpi:
        scale_factor = max(min_dpi / image.width, min_dpi / image.height)
        new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
        image = image.resize(new_size, Image.LANCZOS)
    
    return image

def detect_image_rotation(image: Image.Image) -> int:
    """
    Detect the rotation of text in an image.
    
    Args:
        image: PIL Image object
        
    Returns:
        Rotation angle in degrees (0, 90, 180, or 270)
    """
    try:
        # Using pytesseract's orientation detection
        osd = pytesseract.image_to_osd(image)
        angle = int(osd.split("Rotate: ")[1].split("\n")[0])
        logger.debug(f"Detected rotation angle: {angle} degrees")
        return angle
    except Exception as e:
        logger.warning(f"Could not detect image rotation: {e}")
        return 0  # Assume no rotation

def ocr_with_orientation_correction(image: Image.Image, language: str = settings.OCR_LANGUAGE) -> str:
    """
    Perform OCR with automatic orientation correction.
    
    Args:
        image: PIL Image object
        language: OCR language
        
    Returns:
        Extracted text
    """
    # Preprocess the image
    processed_image = preprocess_image_for_ocr(image)
    
    # Detect orientation
    angle = detect_image_rotation(processed_image)
    
    # Rotate if needed
    if angle != 0:
        processed_image = processed_image.rotate(angle, expand=True)
    
    # Perform OCR
    return perform_ocr(processed_image, language)
