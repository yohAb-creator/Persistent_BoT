import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import List, Tuple, Dict, Any
import re



# Load environment variables
load_dotenv()

# Tree of Thoughts Implementation
class TreeOfThoughts:
    def __init__(self, llm=None, max_depth=3, num_thoughts=4, evaluation_threshold=0.7):
        """
        Initialize a Tree of Thoughts solver.

        Args:
            llm: Language model to use (defaults to GPT-3.5-turbo)
            max_depth: Maximum depth of the tree
            num_thoughts: Number of thoughts to generate at each step
            evaluation_threshold: Threshold for thought evaluation
        """
        self.llm = llm or ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        self.max_depth = max_depth
        self.num_thoughts = num_thoughts
        self.evaluation_threshold = evaluation_threshold

    def generate_thoughts(self, problem: str, previous_thoughts: List[str] = None) -> List[str]:
        """Generate multiple thoughts for a given problem and previous thoughts."""
        previous_thoughts_text = "\n".join(previous_thoughts) if previous_thoughts else "No previous thoughts yet."

        prompt = f"""
        Problem: {problem}

        Previous thoughts:
        {previous_thoughts_text}

        Generate {self.num_thoughts} different possible next thoughts to solve this problem.
        Each thought should be a coherent step towards solving the problem.
        Provide each thought on a new line, numbered 1 through {self.num_thoughts}.
        """

        response = self.llm.invoke(prompt)

        # Extract the numbered thoughts
        thoughts = []
        for line in response.content.split('\n'):
            if re.match(r'^\d+\.', line.strip()):
                thoughts.append(line.strip())

        return thoughts

    def evaluate_thoughts(self, problem: str, thoughts: List[str]) -> List[Tuple[str, float]]:
        """Evaluate the quality of each thought for solving the problem."""
        thoughts_text = "\n".join(thoughts)

        prompt = f"""
        Problem: {problem}

        Thoughts to evaluate:
        {thoughts_text}

        For each thought, evaluate its potential to lead to a solution on a scale from 0.0 to 1.0,
        where 0.0 means the thought is completely irrelevant and 1.0 means it directly solves the problem.
        Provide your evaluation as a list of numbers only, one per line.
        """

        response = self.llm.invoke(prompt)

        # Extract the scores - looking for decimal numbers
        scores = []
        for line in response.content.split('\n'):
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                scores.append(float(match.group(1)))

        # If we couldn't extract scores or have the wrong number, default to 0.5 for each
        if len(scores) != len(thoughts):
            scores = [0.5] * len(thoughts)

        return list(zip(thoughts, scores))

    def solve(self, problem: str) -> str:
        """
        Solve the problem using Tree of Thoughts approach.

        Args:
            problem: Problem statement to solve

        Returns:
            Final solution
        """
        current_depth = 0
        best_thoughts = []

        while current_depth < self.max_depth:
            # Generate new thoughts based on current best thoughts
            thoughts = self.generate_thoughts(problem, best_thoughts)

            if not thoughts:
                break

            # Evaluate thoughts
            evaluated_thoughts = self.evaluate_thoughts(problem, thoughts)

            # Sort thoughts by score
            evaluated_thoughts.sort(key=lambda x: x[1], reverse=True)

            # Keep only thoughts above threshold
            good_thoughts = [t for t, s in evaluated_thoughts if s >= self.evaluation_threshold]

            if not good_thoughts:
                # If no thoughts meet threshold, take the best one
                best_thoughts.append(evaluated_thoughts[0][0])
            else:
                best_thoughts.append(good_thoughts[0])

            current_depth += 1

        # Generate final solution based on the best thoughts
        final_prompt = f"""
        Problem: {problem}

        Thoughts:
        {" ".join(best_thoughts)}

        Based on these thoughts, provide a complete and coherent solution to the problem.
        """

        response = self.llm.invoke(final_prompt)
        return response.content

# Function to extract C++ code from solution text
def extract_cpp_code(solution_text):
    """
    Extract C++ code from the solution text.
    If code blocks are found, return the content of the first C++ code block.
    Otherwise, return the entire solution as it might be just the code without markdown.
    """
    # Try to find code blocks (```cpp ... ```)
    cpp_blocks = re.findall(r'```(?:cpp)?\s*(.*?)```', solution_text, re.DOTALL)

    if cpp_blocks:
        return cpp_blocks[0].strip()

    # If no code blocks found, look for #include which typically indicates C++ code
    if "#include" in solution_text:
        # Try to extract just the code part
        lines = solution_text.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            if "#include" in line:
                start_idx = i
                break

        # Return from the first include to the end
        return '\n'.join(lines[start_idx:])

    # If all else fails, return the entire solution
    return solution_text

# Function to write solution to a C++ file
def write_cpp_solution(solution, filename="solution.cpp"):
    """
    Write the solution to a C++ file.

    Args:
        solution: The solution text which may contain C++ code
        filename: The name of the file to write to

    Returns:
        The path to the created file
    """
    # Extract C++ code from the solution
    cpp_code = extract_cpp_code(solution)

    # Write the code to a file
    with open(filename, 'w') as f:
        f.write(cpp_code)

    print(f"Solution written to {filename}")
    return filename