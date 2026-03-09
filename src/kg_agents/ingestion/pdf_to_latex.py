import os
import time
import json
import shutil

# from urllib import response
import requests
from pathlib import Path
import zipfile

from kg_agents.config.settings import (
    MATHPIX_APP_ID,
    MATHPIX_APP_KEY,
    MATHPIX_PDF_URL,
    MATHPIX_STATUS_URL,
)


class MathpixConversionError(Exception):
    pass


def convert_pdf_to_latex(
    pdf_path: str,
    output_dir: str,
    poll_interval: int = 20,
    timeout: int = 7200,
) -> str:
    """
    Convert a PDF to LaTeX using Mathpix Convert API.

    Args:
        pdf_path (str): Path to input PDF.
        output_dir (str): Directory to store LaTeX output.
        poll_interval (int): Seconds between polling attempts.
        timeout (int): Max time to wait for conversion (seconds).

    Returns:
        str: Path to saved LaTeX file.
    """

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)

    headers = {
        "app_id": MATHPIX_APP_ID,
        "app_key": MATHPIX_APP_KEY,
    }

    print("Uploading PDF to Mathpix...")

    with open(pdf_path, "rb") as f:
        response = requests.post(
            MATHPIX_PDF_URL,
            headers=headers,
            files={"file": f},
            data={"options_json": json.dumps({"conversion_formats": {"latex": True}})},
        )

    if response.status_code != 200:
        raise MathpixConversionError(
            f"Upload failed: {response.status_code} | {response.text}"
        )

    pdf_id = response.json().get("pdf_id")

    if not pdf_id:
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())
        raise MathpixConversionError("No pdf_id returned from Mathpix.")

    print(f"Upload successful. PDF ID: {pdf_id}")
    print("Polling for conversion completion...")

    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            raise MathpixConversionError("Conversion timed out.")

        status_response = requests.get(
            MATHPIX_STATUS_URL.format(pdf_id),
            headers=headers,
        )

        if status_response.status_code != 200:
            raise MathpixConversionError(f"Status check failed: {status_response.text}")

        status_data = status_response.json()
        print(
            f"{status_data.get('num_pages_completed')}/"
            f"{status_data.get('num_pages')} pages done"
        )
        
        status = status_data.get("status")

        if status == "completed":
            print("Conversion completed.")
            break

        elif status == "error":
            raise MathpixConversionError(f"Conversion error: {status_data}")

        print(f"Status: {status} | Waiting {poll_interval}s...")
        time.sleep(poll_interval)

    # Download LaTeX
    # LaTeX zip
    latex_response = requests.get(
        f"https://api.mathpix.com/v3/pdf/{pdf_id}.tex.zip", headers=headers
    )

    if latex_response.status_code != 200:
        raise MathpixConversionError(
            f"LaTeX ZIP download failed: {latex_response.text}"
        )

    pdf_name = Path(pdf_path).stem
    zip_output_path = os.path.join(output_dir, f"{pdf_name}.tex.zip")

    with open(zip_output_path, "wb") as f:
        f.write(latex_response.content)

    print(f"LaTeX ZIP saved to: {zip_output_path}")

    # pdf_name = Path(pdf_path).stem
    # final_tex_path = os.path.join(output_dir, f"{pdf_name}.tex")

    # # Extract .tex directly without preserving internal folders
    # with zipfile.ZipFile(zip_output_path, "r") as zip_ref:
    #     tex_member = None

    #     for member in zip_ref.namelist():
    #         if member.endswith(".tex"):
    #             tex_member = member
    #             break

    #     if not tex_member:
    #         raise MathpixConversionError("No .tex file found inside ZIP.")

    #     # Read .tex file from zip
    #     with zip_ref.open(tex_member) as tex_file:
    #         tex_content = tex_file.read()

    # # Write directly to desired location
    # with open(final_tex_path, "wb") as f:
    #     f.write(tex_content)

    # # Delete ZIP file
    # os.remove(zip_output_path)

    # print(f"LaTeX saved to: {final_tex_path}")

    # return final_tex_path


    pdf_name = Path(pdf_path).stem
    doc_output_dir = os.path.join(output_dir, pdf_name)

    # Ensure clean folder
    if os.path.exists(doc_output_dir):
        shutil.rmtree(doc_output_dir)

    os.makedirs(doc_output_dir, exist_ok=True)

    # Extract full ZIP into that folder
    with zipfile.ZipFile(zip_output_path, "r") as zip_ref:
        zip_ref.extractall(doc_output_dir)

    # Remove ZIP file
    os.remove(zip_output_path)

    print(f"LaTeX project extracted to: {doc_output_dir}")

    return doc_output_dir
