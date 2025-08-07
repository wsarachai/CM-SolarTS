import zipfile
import os

def extract_zip_file(zip_path, extract_to=None):
    """
    Extract files from a ZIP archive.
    
    Args:
        zip_path (str): Path to the ZIP file
        extract_to (str): Directory to extract files to (default: same directory as ZIP file)
    """
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)
    
    print(f"Extracting ZIP file: {zip_path}")
    print(f"Extracting to: {extract_to}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List the contents of the ZIP file
            print("\nFiles in the ZIP archive:")
            file_list = zip_ref.namelist()
            for file_name in file_list:
                print(f"  - {file_name}")
            
            # Extract all files
            print("\nExtracting files...")
            zip_ref.extractall(extract_to)
            print("Extraction completed successfully!")
            
            # Return the list of extracted files
            return [os.path.join(extract_to, file_name) for file_name in file_list]
    
    except zipfile.BadZipFile:
        print("Error: The file is not a valid ZIP archive.")
        return []
    except Exception as e:
        print(f"Error extracting ZIP file: {e}")
        return []

if __name__ == "__main__":
    # Path to the ZIP file
    zip_file_path = "data/download.grib"
    
    # Extract the ZIP file
    extracted_files = extract_zip_file(zip_file_path)
    
    if extracted_files:
        print("\nExtracted files:")
        for file_path in extracted_files:
            print(f"  - {file_path}")