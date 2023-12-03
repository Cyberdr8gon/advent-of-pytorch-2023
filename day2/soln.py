import torch

import argparse
import re

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

# Required positional argument
parser.add_argument('filename', type=str,
                    help='games record file.')

# ordered in red, green, blue cubes
TEST_CUBE_COUNT = torch.tensor([12, 13, 14])

def main(args):
    if len(args.filename) == 0:
        parser.error("must supply a calibration values filename.")

    cube_counts = []
    longest_game = 0

    with open(args.filename) as file:
        # First parse the input.
        # game num: game 1; game 2; game 3
        for line_unstripped in file:
            line = line_unstripped.rstrip()
            line_split = line.split(":")
            game_number = int(re.findall(r'\d+', line_split[0])[0])
            game_split = line_split[1].split(";")

            sub_games = []

            for game in game_split:
                game_result = []

                red_results = re.findall(r'(\d+) red', game)
                if len(red_results) > 0:
                    game_result.append(int(red_results[0]))
                else:
                    game_result.append(0)

                green_results = re.findall(r'(\d+) green', game)
                if len(green_results) > 0:
                    game_result.append(int(green_results[0]))
                else:
                    game_result.append(0)

                blue_results = re.findall(r'(\d+) blue', game)
                if len(blue_results) > 0:
                    game_result.append(int(blue_results[0]))
                else:
                    game_result.append(0)

                sub_games.append(game_result)
            
            longest_game = max(longest_game, len(sub_games))
            cube_counts.append(sub_games)

    # add padding games which always succeed so all are the right length.
    cube_counts = [ game_result + [[0, 0, 0] for _ in range(longest_game - len(game_result))] for game_result in cube_counts]

    # now tensorify.
    cube_counts = torch.tensor(cube_counts)


    # Now check which subgames violated the limits by first reducing the subgame dims, then game dim.
    is_valid = cube_counts <= TEST_CUBE_COUNT

    is_valid = torch.all(is_valid, dim=-1)
    is_valid = torch.all(is_valid, dim=-1)

    # Now multiply by game index.
    game_indices = torch.arange(1, is_valid.shape[0] + 1)

    sum_of_idx = torch.sum(is_valid * game_indices)

    print("sum_of_idx:", sum_of_idx.numpy())

    # Part 2, powers of cubes.

    min_cubes_per_game, _ = torch.max(cube_counts, dim=-2)

    min_cubes_powers = torch.prod(min_cubes_per_game, dim=-1)

    sum_of_min_cubes_powers = torch.sum(min_cubes_powers)

    print("sum_of_min_cubes_powers:", sum_of_min_cubes_powers.numpy())


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
