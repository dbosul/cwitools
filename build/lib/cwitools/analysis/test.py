"""Adding docstring to test"""
import argparse

def main():
    print("This is working so far...")
    parser = argparse.ArgumentParser(description="Test of console entry point.")
    parser.add_argument("word", type=str, help="Enter a word.", default=None )
    args = parser.parse_args()
    print(args.word)
