OCR reader using PaddleOCR

Usage:

1. Create a Python virtualenv and install dependencies. PaddleOCR requires paddlepaddle; on Windows install the suitable paddlepaddle wheel (CPU) first. Example:

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
# install paddlepaddle per instructions (choose matching python/cpu)
pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
pip install -r requirements.txt
```

2. Run the reader against `ocr_jobs/main_tmp` (default) or any folder with crops:

```powershell
python ocr_reader.py ocr_jobs/main_tmp --out results.json --visualize
```

Output:
- `results.json` — consolidated JSON with OCR results per image path
- `*_ocr.jpg` — optional visualization images when `--visualize` is used

Notes:
- The script tries to be tolerant across PaddleOCR versions. If import fails, ensure `paddleocr` and `paddlepaddle` are installed.
- On large batches, consider adding batching or simple concurrency.