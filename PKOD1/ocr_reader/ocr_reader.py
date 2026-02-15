import os
import sys
import json
import logging
from glob import glob
from rapidocr_onnxruntime import RapidOCR

def find_images(path):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for pat in patterns:
        files.extend(glob(os.path.join(path, "**", pat), recursive=True))
    files = sorted(files)
    return files

def main():
    # 1. Setup Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # 2. Hardcoded Paths
    indir = r"C:\Users\Adarsh\OneDrive\Documents\PKOD\ocr_jobs"
    out_file = r"C:\Users\Adarsh\OneDrive\Documents\PKOD\ocr_results.json"

    # 3. Initialize RapidOCR
    # It automatically downloads the best models (same ones Paddle uses)
    try:
        engine = RapidOCR()
        logger.info("RapidOCR initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to init RapidOCR: {e}")
        sys.exit(2)

    # 4. Check Input Directory
    if not os.path.isdir(indir):
        logger.error(f'Input directory not found: {indir}')
        sys.exit(2)

    files = find_images(indir)
    if not files:
        logger.warning(f'No images found in {indir}')
        sys.exit(0)

    logger.info(f'Found {len(files)} images. Starting processing...')

    results = []
    
    # 5. Process Images
    for f in files:
        short_name = os.path.basename(f)
        logger.info(f'Processing: {short_name}')
        
        try:
            # RapidOCR returns: result, elapse_list
            # result is a list of [box, text, score]
            ocr_result, _ = engine(f)
            
            if ocr_result is None:
                logger.warning(f"No text found in {short_name}")
                continue

            captured_texts = []
            confidences = []

            for item in ocr_result:
                # item structure: [ [[x,y]...], "text", "0.98" ]
                text = item[1]
                score = float(item[2])
                
                captured_texts.append(text)
                confidences.append(score)
            
            combined_text = " ".join(captured_texts)
            
            if confidences:
                avg_conf = sum(confidences) / len(confidences)
            else:
                avg_conf = 0.0

            result_entry = {
                "image": short_name,
                "full_path": f,
                "text": combined_text,
                "confidence": round(avg_conf, 2)
            }
            results.append(result_entry)
            
        except Exception as e:
            logger.warning(f'Skipped {short_name}: {e}')

    # 6. Save Results
    with open(out_file, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    logger.info(f'Done! Results saved to {out_file}')

if __name__ == '__main__':
    main()