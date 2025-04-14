

#compare txt files
import numpy as np
import difflib
import sys
dfile1="embedding_outputs/embedding_output_hf.txt"
dfile2="embedding_outputs/embedding_output_jax.txt"
def compare_files(file1, file2):
    """
    Compare two text files and print the differences.
    """
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        if len(lines1) != len(lines2):
            print(f"Files {file1} and {file2} have different number of lines.")
            return
        for i, (line1, line2) in enumerate(zip(lines1, lines2)):
            if line1 != line2:
                print(f"Difference in line {i + 1}:")
                
    # Use difflib to compare the files
compare_files(dfile1, dfile2)