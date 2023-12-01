import torch

import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

# Required positional argument
parser.add_argument('filename', type=str,
                    help='Calibration values file.')

NUMBER_STRINGS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

def main(args):
    if len(args.filename) == 0:
        parser.error("must supply a calibration values filename.")

    codes = []

    # parse codes as ascii
    with open(args.filename) as file:
        for line_unstripped in file:
            line = line_unstripped.rstrip()
            # CHEESE. can't do strings in pytorch so we switch them here in python.
            line = line.lower()
            for i, el in enumerate(NUMBER_STRINGS):
                line = line.replace(el, el[0] + str(i+1) + el[1:])
            
            code = []
            for el in line:
                code.append(ord(el))

            codes.append(code)

    # pad to allow for tensorifying
    max_len = len(max(codes, key=len))
    num_lines = len(codes)

    codes = [code + [0 for _ in range(max_len - len(code))] for code in codes]

    codes = torch.tensor(codes)

    # Now the real solution begins.
    # First, find which elements are numbers.
    below_number_mask = torch.where(codes > 47, True, False)
    above_number_mask = torch.where(codes < 58, True, False)

    # Then mask out everything that isn't a number.
    cleaned_codes = codes * (below_number_mask & above_number_mask).int().float()

    # Then, find all indicies that are non-zero.
    indicies = cleaned_codes.nonzero()

    # Then, find the min and the max index for each row.
    min_values = []
    max_values = []
    for i in range(num_lines):
        row_idxs = indicies[indicies[:, 0] == i][:, 1]
        
        first_idx = row_idxs[0]
        last_idx = row_idxs[-1]

        min_values.append(cleaned_codes[i, first_idx])
        max_values.append(cleaned_codes[i, last_idx])
    
    min_values = torch.tensor(min_values) - 48
    max_values = torch.tensor(max_values) - 48

    calibration_values = min_values * 10 + max_values

    print("calibration_values", calibration_values)

    print("sum of calibration_values", torch.sum(calibration_values))



        



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
