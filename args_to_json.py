"""
A helper utility to convert a list of function arguments with
their default values into a corresponding JSON file snippet
"""
import ast
import json
import sys
import os

def parse_arguments(arg_list):
    """
    Parses a comma-separated list of function arguments with default values.
    """
    defaults = {}
    for arg in arg_list.split(','):
        arg = arg.strip()
        if '=' in arg:
            name, value = arg.split('=', 1)
            name = name.strip()
            value = value.strip()
            try:
                # Use ast.literal_eval to safely evaluate the value
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            defaults[name] = value
    return defaults

def append_json_to_file(arg_list, input_file):
    """
    Appends the JSON representation of function arguments with default values to the input file.
    """
    defaults = parse_arguments(arg_list)
    
    with open(input_file, 'a') as f:
        f.write("\n\n")
        json.dump(defaults, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.isfile(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    
    with open(input_file, 'r') as f:
        arg_list = f.read().strip()
    
    append_json_to_file(arg_list, input_file)

