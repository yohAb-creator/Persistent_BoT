import subprocess
import tempfile
import os
import re
from typing import Tuple


class SolutionValidator:
    """
    Validates a C++ solution by simulating a two-file compilation environment
    similar to competitive programming judges like DMOJ.
    """

    def _extract_examples(self, problem_statement: str) -> list:
        """Extracts input/output examples from the problem text."""
        examples = []
        pattern = re.compile(r"`beechtree\((.*?)\)` should return `(.*?)`")
        matches = pattern.findall(problem_statement)

        for match in matches:
            inputs_str, output_str = match
            try:
                expected_output = ' '.join(output_str.strip('[]').replace(',', '').split())

                # A more robust parser for the input string
                cleaned_input = re.sub(r'\[-?\d+(,\s*-?\d+)*\]', lambda m: m.group().replace(',', ' '), inputs_str)
                cleaned_input = cleaned_input.replace(',', ' ')
                parts = cleaned_input.split()
                N, M = parts[0], parts[1]
                P_array = ' '.join(parts[2:2 + int(N)])
                C_array = ' '.join(parts[2 + int(N):])

                # Format for stdin
                input_data = f"{N} {M}\n{P_array}\n{C_array}"
                examples.append({"input": input_data, "output": expected_output})
            except Exception as e:
                print(f"Warning: Could not parse example: {match} due to {e}")

        return examples

    def _create_mock_header(self, header_path: str):
        """Creates a mock header file declaring the required function."""
        header_content = """
#pragma once
#include <vector>

std::vector<int> beechtree(int N, int M, std::vector<int> P, std::vector<int> C);
"""
        with open(header_path, "w") as f:
            f.write(header_content)

    def _create_mock_grader(self, grader_path: str):
        """Creates a mock grader file with a main function that reads from stdin."""
        grader_content = """
#include <iostream>
#include <vector>
#include "beechtree.h" // Includes the function declaration

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    int N, M;
    std::cin >> N >> M;
    std::vector<int> P(N), C(N);
    for (int i = 0; i < N; ++i) std::cin >> P[i];
    for (int i = 0; i < N; ++i) std::cin >> C[i];

    std::vector<int> result = beechtree(N, M, P, C);

    for (int i = 0; i < result.size(); ++i) {
        std::cout << result[i] << (i == result.size() - 1 ? "" : " ");
    }
    std::cout << std::endl;

    return 0;
}
"""
        with open(grader_path, "w") as f:
            f.write(grader_content)

    def validate_cpp_solution(self, solution_code: str, problem_statement: str) -> Tuple[bool, str]:
        """
        Compiles and runs the C++ code against all extracted examples using a
        mock grader and header to simulate a real judge environment.
        """
        examples = self._extract_examples(problem_statement)
        if not examples:
            return True, "No examples found to validate against."

        with tempfile.TemporaryDirectory() as temp_dir:
            solution_path = os.path.join(temp_dir, "solution.cpp")
            header_path = os.path.join(temp_dir, "beechtree.h")
            grader_path = os.path.join(temp_dir, "grader.cpp")
            executable_path = os.path.join(temp_dir, "solution_executable")

            # Write the necessary files
            with open(solution_path, "w") as f:
                # Add common includes if they are missing from the generated code
                if "#include" not in solution_code:
                    solution_code = "#include <vector>\n" + solution_code
                f.write(solution_code)

            self._create_mock_header(header_path)
            self._create_mock_grader(grader_path)

            # --- Compilation Step ---
            # Compile both the grader and the solution file together
            compile_process = subprocess.run(
                ["g++", "-std=c++17", "-o", executable_path, grader_path, solution_path],
                capture_output=True, text=True
            )
            if compile_process.returncode != 0:
                return False, f"Compilation failed: {compile_process.stderr}"

            # --- Execution and Validation Step ---
            for i, example in enumerate(examples):
                try:
                    run_process = subprocess.run(
                        [executable_path],
                        input=example["input"],
                        capture_output=True, text=True, timeout=5
                    )
                    if run_process.returncode != 0:
                        return False, f"Example {i + 1} failed: Runtime error: {run_process.stderr}"

                    actual_output = run_process.stdout.strip()
                    if actual_output != example["output"]:
                        return False, f"Example {i + 1} failed: Wrong answer. Expected '{example['output']}', got '{actual_output}'"
                except subprocess.TimeoutExpired:
                    return False, f"Example {i + 1} failed: Timeout."

        return True, "All examples passed."
