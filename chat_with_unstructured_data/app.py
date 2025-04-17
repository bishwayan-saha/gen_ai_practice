from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
import os, base64

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "docs", "attention.pdf")

chunks = partition_pdf(
    filename=file_path,
    infer_table_structure=True, # Infer table structure
    strategy='hi-res', # Mandatory for table extraction

    extract_image_block_types=['Image'],
    extract_image_block_to_payload=True, # Extract base64 version of image and metadata
    chunking_strategy='by_title',
    max_characters=10000,
    combine_text_under_n_chars=2000,
    new_after_n_chars=6000
)

tables = []
texts = []
images_b64 = []

for chunk in chunks:
    if 'Table' in str(type(chunk)):
        tables.append(chunk)
    if 'Composite' in str(type(chunk)):
        texts.append(chunk)
        chunk_elements = chunk.metadata.orig_elements
        for element in chunk_elements:
            if 'Image' in str(type(chunk)):
                images_b64.append(element.metadata.image_base64)

print(tables[0])


