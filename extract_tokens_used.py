import os
import re
import json
import statistics
import argparse
from pathlib import Path


def extract_token_usage(log_directory):
    """
    Searches for team orchestration log files in the specified directory,
    extracts "Total token usage" information, and calculates statistics.
    Saves the output in the same directory.

    Args:
        log_directory: Path to the directory containing log files
    """
    # Dictionary to store filename -> token usage mapping
    token_data = {}

    # Create path to the log directory
    log_dir_path = Path(log_directory)

    # Find only the team orchestration log files in the specified directory
    # by looking for files containing "conversation_log_agent_team_orchestration" in their names
    text_files = list(
        log_dir_path.glob("*conversation_log_agent_team_orchestration*.txt")
    )

    # Regular expression to find token usage
    token_pattern = re.compile(r"Total token usage: Tokens: (\d+)")

    print(
        f"Found {len(text_files)} team orchestration files to process in {log_directory}/"
    )

    # Process each file
    for file_path in text_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                file_content = file.read()

                # Search for the token usage pattern
                match = token_pattern.search(file_content)
                if match:
                    token_count = int(match.group(1))
                    token_data[file_path.name] = token_count
                    print(f"Found token usage in {file_path.name}: {token_count}")
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    # Calculate statistics if we have data
    statistics_data = {}
    if token_data:
        token_values = list(token_data.values())
        statistics_data = {
            "mean": statistics.mean(token_values),
            "std_dev": statistics.stdev(token_values) if len(token_values) > 1 else 0,
            "min": min(token_values),
            "max": max(token_values),
            "total": sum(token_values),
            "count": len(token_values),
        }

    # Save the results to tokens_used.txt in the specified directory
    output_path = log_dir_path / "tokens_used.txt"
    with open(output_path, "w", encoding="utf-8") as output_file:
        # Write a readable format
        output_file.write(
            f"Token Usage Summary for Team Orchestration Logs (from {log_directory}/):\n"
        )
        output_file.write("===================\n\n")

        for filename, tokens in sorted(token_data.items()):
            output_file.write(f"{filename}: {tokens} tokens\n")

        # Write statistics
        if statistics_data:
            output_file.write("\nStatistics:\n")
            output_file.write("===========\n")
            output_file.write(
                f"Total files with token data: {statistics_data['count']}\n"
            )
            output_file.write(f"Total tokens: {statistics_data['total']}\n")
            output_file.write(f"Mean tokens per file: {statistics_data['mean']:.2f}\n")
            output_file.write(f"Standard deviation: {statistics_data['std_dev']:.2f}\n")
            output_file.write(f"Minimum tokens: {statistics_data['min']}\n")
            output_file.write(f"Maximum tokens: {statistics_data['max']}\n")

        # Also write as JSON for easier processing
        output_file.write("\n\n# JSON format for programmatic use:\n")
        output_file.write(
            json.dumps(
                {"token_data": token_data, "statistics": statistics_data}, indent=2
            )
        )

    print(f"Results saved to {output_path}")
    print(f"Found token usage data in {len(token_data)} out of {len(text_files)} files")

    if statistics_data:
        print("\nStatistics:")
        print(f"Mean tokens per file: {statistics_data['mean']:.2f}")
        print(f"Standard deviation: {statistics_data['std_dev']:.2f}")
        print(f"Min tokens: {statistics_data['min']}")
        print(f"Max tokens: {statistics_data['max']}")
        print(f"Total tokens: {statistics_data['total']}")

    return {"token_data": token_data, "statistics": statistics_data}


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Extract token usage from team orchestration log files"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="default_experiment",
        help="Directory containing log files (default: default_experiment)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Run the extraction with the specified directory
    extract_token_usage(args.directory)
