import os
import zipfile

file_path = "dataset/cams-gridded-solar-radiation.zip"  # Default file path

# Check if the file exists
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found")
    exit(1)

# Try to open as a ZIP file
try:
    print(f"Opening {file_path} as a ZIP file...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        print("\nFiles in the archive:")
        file_list = zip_ref.namelist()
        for file in file_list:
            file_info = zip_ref.getinfo(file)
            print(f"- {file} (Size: {file_info.file_size} bytes)")
        
        # Extract all files
        print("\nExtracting files...")
        zip_ref.extractall(".")
        print("Files extracted successfully!")
        
except zipfile.BadZipFile:
    print(f"Error: {file_path} is not a valid ZIP file")
    
    # Try to get basic file information
    file_size = os.path.getsize(file_path)
    print(f"\nFile size: {file_size} bytes")
    
    # Try to read first few bytes to check file type
    with open(file_path, 'rb') as f:
        header = f.read(16)
        print(f"File header: {header}")
        
except Exception as e:
    print(f"Error: {e}")
    
    # Try to get basic file information
    file_size = os.path.getsize(file_path)
    print(f"\nFile size: {file_size} bytes")
    
    # Try to read first few bytes to check file type
    with open(file_path, 'rb') as f:
        header = f.read(16)
        print(f"File header: {header}")