import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import math

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

# Required positional argument
parser.add_argument('filename', type=str,
                    help='engine schematic file.')
# idea to solve:
# Find all clusters of numbers and their idxs in image space.
# make an image of 1s of symbols
# use an all 1s convolution filter to find all valid number spots (morphological opening).
# gather_nd lookup all the bool values of adjacent or not for each number.
# If any are true, its a valid number. 
# sum all valid numbers



def main(args):
    if len(args.filename) == 0:
        parser.error("must supply a calibration values filename.")

    symbol_grid = []
    gear_idxs = []
    numbers = []
    number_indicies = []

    with open(args.filename) as file:
        # First parse the input.
        for i, line_unstripped in enumerate(file):
            line = line_unstripped.rstrip()

            symbol_grid.append([])

            current_number = []
            current_number_indicies = []

            last_was_digit = False

            for j, c in enumerate(line):
                # First, build the symbol grid.
                if c != '.' and not c.isalpha() and not c.isdigit():
                    symbol_grid[i].append(True)
                    gear_idxs.append([i, j])
                else:
                    symbol_grid[i].append(False)

                # Next, accumulate any numbers we find:
                if c.isdigit():
                    current_number.append(c)
                    current_number_indicies.append((i, j))
                    last_was_digit = True
                elif last_was_digit:
                    last_was_digit = False
                    numbers.append(int("".join(current_number)))
                    number_indicies.append(current_number_indicies)
                    current_number = []
                    current_number_indicies = []
            if last_was_digit:
                numbers.append(int("".join(current_number)))
                number_indicies.append(current_number_indicies)
                current_number = []
                current_number_indicies = []

        # Next convert to tensors.
        symbol_grid = torch.tensor(symbol_grid)
        numbers = torch.tensor(numbers)

        number_indicies = [[[x, y] for x, y in el] for el in number_indicies]
        num_idx_lens = torch.tensor([len(indicies) for indicies in number_indicies])
        max_idx_len = max(num_idx_lens)
        number_indicies = [torch.tensor(el) for el in number_indicies]
        number_indicies = torch.stack([F.pad(indicies, (0, 0, 0, max_idx_len - len(indicies))) for indicies in number_indicies])

        # Also build a mask for each idx for those that are valid.
        valid_idx_mask = torch.arange(max_idx_len).expand(len(num_idx_lens), max_idx_len)
        valid_idx_mask = valid_idx_mask < num_idx_lens.unsqueeze(1)

        # Turn the symbol grid into a 1 batch size, 1 channel, channels_first image 
        symbol_grid = symbol_grid.view(1, 1, symbol_grid.shape[0], symbol_grid.shape[1]).to(torch.float32)

        # Setup all ones convolution to find all pixels adjacent to symbols.
        conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        nn.init.constant_(conv.weight, 1)

        # Apply the convolution.
        is_symbol_adjacent = conv(symbol_grid)
        is_symbol_adjacent = 0 < is_symbol_adjacent

        # Next, lookup which row has valid indicies
        rows_lookup = number_indicies[..., 0].flatten()
        cols_lookup = number_indicies[..., 1].flatten()

        indexed_values = is_symbol_adjacent[0, 0, rows_lookup, cols_lookup]

        indexed_values = indexed_values.view(number_indicies.shape[:-1])

        # Next, apply the mask to ensure we get no random trues.
        indexed_values = torch.logical_and(indexed_values, valid_idx_mask)

        # Then simplify per row.
        indexed_values = torch.any(indexed_values, dim=-1)

        # Finally, multiply with our numbers and add them all up.
        valid_numbers = indexed_values.to(torch.float32) * numbers
        sum_of_valid_numbers = torch.sum(valid_numbers)

        print("sum_of_valid_numbers", sum_of_valid_numbers)

        # Next, lets identify the sum of the gear ratios. To do so, we will first make
        # a pairwise distance matrix between each number and each gear.
        gear_idxs = torch.tensor(gear_idxs)
        distance_matrix, _ = torch.min(torch.linalg.norm(gear_idxs.view(1, gear_idxs.shape[0], 1, 2).float() - number_indicies.unsqueeze(1).float(), dim=-1), dim=-1)

        # Then we will build an adjacency matrix, and use this to detect which are 
        # "true_gears" because they have 2 and only 2 adjacent numbers.
        # Indexing is [number, gear]
        adjacency_matrix = distance_matrix < 1.99

        true_gears = torch.sum(adjacency_matrix.int(), dim=0) == 2

        # Then we use the true gears to look up their adjacent number indexes in the 
        # adjacency matrix.
        gear_ratio_indicies = torch.transpose(adjacency_matrix[:, true_gears], 0, 1)
        
        # Next, tile the numbers so that we can use boolean indexing from the gear ratio
        # indicies.
        numbers_expanded = numbers.view(1, -1).expand(gear_ratio_indicies.size(0), -1)

        # NOTE(sbateman): The above transposes must be here for an incredibly dumb reason.
        # Unfortunately, pytorch doesn't respect the ordering of the input indicies and 
        # always returns them in the row ordering regardless of how the input indicies 
        # were provided. This is likely more efficient from a slicing point of view, however,
        # it makes recovering pairs of searched for numbers subtly difficult if you
        # don't make the pairing axis the row axis.
        # Thus, we must make gears the primary axis here through a transpose so it respects
        # the ordering of the gear indicies during boolean indexing.
        gear_ratios = numbers_expanded[gear_ratio_indicies]

        # Reshape to have 2 numbers per gear.
        gear_ratios = gear_ratios.view(-1, 2)
        gear_ratios = torch.prod(gear_ratios, dim=-1)

        sum_of_gear_ratios = torch.sum(gear_ratios)

        print("sum_of_gear_ratios", sum_of_gear_ratios)



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
