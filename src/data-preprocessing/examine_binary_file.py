#!/usr/bin/env python3
"""
Script to examine binary file content to determine its actual format
"""

import os
import binascii

def examine_binary_file(file_path):
    """Examine the binary content of a file to determine its format"""
    print(f"Examining binary file: {file_path}")
    print(f"File size: {os.path.getsize(file_path)} bytes")
    
    with open(file_path, 'rb') as f:
        # Read first 100 bytes to identify file type
        header = f.read(100)
        
        print("\nFirst 100 bytes (hex):")
        print(binascii.hexlify(header).decode('ascii'))
        
        print("\nFirst 100 bytes (ASCII representation):")
        ascii_repr = ''
        for i, byte in enumerate(header):
            if 32 <= byte <= 126:  # Printable ASCII
                ascii_repr += chr(byte)
            else:
                ascii_repr += '.'
            if (i + 1) % 20 == 0:
                ascii_repr += '\n'
        print(ascii_repr)
        
        # Check for common file signatures
        print("\nFile signature analysis:")
        
        # Check for ZIP signature
        if header.startswith(b'PK\x03\x04'):
            print("  [+] ZIP file signature detected")
        elif header.startswith(b'PK\x05\x06') or header.startswith(b'PK\x06\x06'):
            print("  [+] ZIP file signature detected (empty archive)")
        
        # Check for GRIB signature
        elif header.startswith(b'GRIB'):
            print("  [+] GRIB file signature detected")
        
        # Check for NetCDF signature
        elif header.startswith(b'CDF'):
            print("  [+] NetCDF file signature detected")
        
        # Check for common text file signatures
        elif header.startswith(b'{') or header.startswith(b'['):
            print("  [+] JSON-like structure detected")
        
        elif header.startswith(b'<?xml'):
            print("  [+] XML file detected")
        
        elif header.startswith(b'\x89PNG'):
            print("  [+] PNG image detected")
        
        elif header.startswith(b'%PDF'):
            print("  [+] PDF document detected")
        
        else:
            print("  ? Unknown file format")
        
        # Check if it's a text file
        try:
            with open(file_path, 'r', encoding='utf-8') as text_file:
                first_line = text_file.readline()
                print(f"\nFirst line as text: {repr(first_line)}")
        except UnicodeDecodeError:
            print("\nFile does not appear to be UTF-8 text")
        except Exception as e:
            print(f"\nError reading as text: {e}")

if __name__ == "__main__":
    grib_file = "dataset/derived-utci-historical.grib"
    if os.path.exists(grib_file):
        examine_binary_file(grib_file)
    else:
        print(f"File not found: {grib_file}")