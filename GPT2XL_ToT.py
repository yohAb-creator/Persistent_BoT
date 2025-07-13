"""
C++ Code Cleaner for GPT-2 Output

This module contains improved functions for creating clean C++ code
from GPT-2 output, particularly for competitive programming solutions.
"""

import re
from GPT2_TOT import GPT2CompetitiveProgrammingToT
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Tuple, Dict, Any, Optional
import re
import json
import os
import time

# Initialize GPT2
def initialize_gpt2(model_name="gpt2-xl"):
    """
    Initialize GPT-2 model and tokenizer.

    Args:
        model_name: Name of the GPT-2 model variant to use
                   Options: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"

    Returns:
        Tuple containing (model, tokenizer)
    """
    print(f"Loading {model_name} model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Add padding token to tokenizer if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"{model_name} loaded successfully!")
    return model, tokenizer
# GPT2 Prompt Wrapper
def generate_text_with_gpt2(
    prompt: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    max_new_tokens: int = 200,
    num_return_sequences: int = 1,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2
) -> List[str]:
    """
    Generate text using GPT-2 based on a prompt.

    Args:
        prompt: The text prompt to generate from
        model: The GPT-2 model
        tokenizer: The GPT-2 tokenizer
        max_new_tokens: Maximum number of new tokens to generate
        num_return_sequences: Number of sequences to generate
        temperature: Higher values produce more diverse results
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repeating tokens

    Returns:
        List of generated text sequences
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode the generated text
    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        # Remove the original prompt from the generated text
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        generated_texts.append(text)

    return generated_texts

def extract_cpp_code(solution_text):
    """
    Extract clean C++ code from the solution text.
    Handles various formats and cleans up the output to produce valid C++ code.

    Args:
        solution_text: The text containing a C++ solution (possibly with extra content)

    Returns:
        Clean, properly formatted C++ code
    """
    # Try to find code blocks (```cpp ... ```)
    cpp_blocks = re.findall(r'```(?:cpp)?\s*(.*?)```', solution_text, re.DOTALL)

    if cpp_blocks:
        code = cpp_blocks[0].strip()
    else:
        # If no code blocks found, look for #include which typically indicates C++ code
        if "#include" in solution_text:
            # Try to extract just the code part
            lines = solution_text.split('\n')
            start_idx = 0
            for i, line in enumerate(lines):
                if "#include" in line:
                    start_idx = i
                    break

            # Get from the first include to the end
            code = '\n'.join(lines[start_idx:])
        else:
            # If all else fails, return the entire solution
            code = solution_text

    # Clean up the code

    # 1. Remove any "Output:" sections and everything after them
    output_match = re.search(r'\bOutput:\s*', code, re.IGNORECASE)
    if output_match:
        code = code[:output_match.start()]

    # 2. Find and remove any nested or secondary #include statements
    # This handles cases where GPT-2 generates multiple solutions
    lines = code.split('\n')
    first_include_idx = -1
    include_indices = []

    for i, line in enumerate(lines):
        if "#include" in line:
            if first_include_idx == -1:
                first_include_idx = i
            else:
                include_indices.append(i)

    # If we found multiple #include statements, keep only the first solution
    if include_indices:
        # Find the first main after the first include
        first_main_idx = -1
        for i in range(first_include_idx, len(lines)):
            if "int main" in lines[i]:
                first_main_idx = i
                break

        # Find the closing brace of the first main
        if first_main_idx != -1:
            brace_count = 0
            end_idx = len(lines)

            for i in range(first_main_idx, len(lines)):
                if '{' in lines[i]:
                    brace_count += lines[i].count('{')
                if '}' in lines[i]:
                    brace_count -= lines[i].count('}')
                    if brace_count <= 0:
                        end_idx = i + 1
                        break

            # Keep only the first solution
            lines = lines[:end_idx]
            code = '\n'.join(lines)

    # 3. Look for alternative solutions and remove them
    alt_solution_patterns = [
        r'\bSolution with\b.*',
        r'\bAlternative solution\b.*',
        r'\bAnother approach\b.*'
    ]

    for pattern in alt_solution_patterns:
        match = re.search(pattern, code, re.IGNORECASE | re.DOTALL)
        if match:
            code = code[:match.start()]

    # 4. Remove any input/output example comments
    lines = code.split('\n')
    cleaned_lines = []

    skip_mode = False
    for line in lines:
        # Skip lines with Example, Input, Output keywords in comments
        if re.search(r'^\s*(?://|/\*|\*)\s*(?:Example|Sample|Test|Input|Output)(?:Input|Output)?', line, re.IGNORECASE):
            skip_mode = True
            continue

        # Exit skip mode if we see a non-comment line
        if skip_mode and not re.match(r'^\s*(?://|/\*|\*)', line):
            skip_mode = False

        # Only add lines when not in skip mode
        if not skip_mode:
            cleaned_lines.append(line)

    code = '\n'.join(cleaned_lines)

    # 5. Remove trailing comments about time complexity, explanations, etc.
    code = re.sub(r'\n\s*//\s*[Tt]ime [Cc]omplexity.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n\s*//\s*[Ss]pace [Cc]omplexity.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n\s*//\s*[Ee]xplanation:.*$', '', code, flags=re.MULTILINE)

    # 6. Fix formatting (indentation, whitespace, etc.)
    lines = code.split('\n')
    formatted_lines = []

    inside_main = False
    indent_level = 0

    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        cleaned_line = line.strip()

        # Track main function and adjust indentation
        if "int main" in cleaned_line:
            inside_main = True
            formatted_lines.append(cleaned_line)
            if '{' in cleaned_line:
                indent_level = 1
            continue

        # Adjust indentation based on braces
        if '{' in cleaned_line:
            # Add line then increase indent
            if inside_main:
                formatted_lines.append("    " * indent_level + cleaned_line)
            else:
                formatted_lines.append(cleaned_line)
            indent_level += 1
            continue

        if '}' in cleaned_line:
            # Decrease indent then add line
            indent_level = max(0, indent_level - 1)
            if inside_main and indent_level > 0:
                formatted_lines.append("    " * indent_level + cleaned_line)
            else:
                formatted_lines.append(cleaned_line)
                inside_main = False
            continue

        # Add regular line with appropriate indentation
        if inside_main and indent_level > 0:
            formatted_lines.append("    " * indent_level + cleaned_line)
        else:
            formatted_lines.append(cleaned_line)

    code = '\n'.join(formatted_lines)

    # 7. Ensure the main function is properly closed with a return statement
    if "int main()" in code or "int main(" in code:
        # Check if it already has a return statement
        if "return" not in code:
            # Add return 0 before the last closing brace or at the end
            last_brace = code.rfind('}')
            if last_brace != -1:
                code = code[:last_brace] + "\n    return 0;\n}" + code[last_brace + 1:]
            else:
                code = code + "\n    return 0;\n}"

        # Check for proper closing of main function
        if code.count('{') > code.count('}'):
            code += "\n}"

    return code.strip()


def write_cpp_solution(solution, filename=None):
    """
    Write the solution to a C++ file.

    Args:
        solution: The solution text which may contain C++ code
        filename: The name of the file to write to (optional)

    Returns:
        The path to the created file
    """
    # Extract C++ code from the solution
    cpp_code = extract_cpp_code(solution)

    # If no filename provided, create one based on first few characters
    if not filename:
        # Use first few words of the solution to create a filename
        words = re.findall(r'\w+', cpp_code[:100])
        if words:
            filename = f"{'_'.join(words[:3])[:30]}.cpp"
        else:
            filename = "solution.cpp"

    # Ensure filename ends with .cpp
    if not filename.endswith('.cpp'):
        filename += '.cpp'

    # Write the code to a file
    with open(filename, 'w') as f:
        f.write(cpp_code)

    print(f"Solution written to {filename}")
    return filename


def simple_gpt2_coding(problem, model, tokenizer):
    """
    Generate a structured C++ solution to a competitive programming problem using GPT-2.
    Uses a carefully structured prompt to guide the model toward cleaner code.

    Args:
        problem: The problem statement
        model: The GPT-2 model
        tokenizer: The GPT-2 tokenizer

    Returns:
        A clean C++ solution to the problem
    """
    # Parse the problem to extract key information
    problem_lines = [line.strip() for line in problem.strip().split('\n') if line.strip()]

    # Create a structured prompt that will guide the model
    prompt = f"""
// Problem: Write a C++ program to solve the following problem:
// {' '.join(problem_lines)}
//
// Provide only the solution code, no explanations or examples.

#include <bits/stdc++.h>
using namespace std;

int main() {{
"""

    # Generate the solution with lower temperature for more focused output
    generated_text = generate_text_with_gpt2(
        prompt, model, tokenizer,
        max_new_tokens=150,  # Limit token count to avoid rambling
        temperature=0.2,  # Lower temperature for more focused output
        top_p=0.85,  # Focus on more likely tokens
        repetition_penalty=1.2  # Discourage repetition
    )[0]

    # Ensure proper closing of the solution
    full_solution = prompt + generated_text

    # Check if solution is complete (has a return statement and closing brace)
    if "return 0" not in full_solution and "return 0;" not in full_solution:
        full_solution += "\n    return 0;\n}"

    # Make sure the main function is properly closed
    if full_solution.count('{') > full_solution.count('}'):
        full_solution += "\n}"

    return full_solution


def solve_competitive_problem(problem_statement, problem_name=None, model_name="gpt2-xl", use_simplified=True):
    """
    Solve a competitive programming problem and save the solution to a C++ file.

    Args:
        problem_statement: The full problem statement
        problem_name: Optional name for the problem (used for the filename)
        model_name: The GPT-2 model to use
        use_simplified: Whether to use the simplified coding approach

    Returns:
        Tuple containing (solution_text, cpp_filename)
    """
    print(f"Initializing GPT-2 solver with {model_name}...")

    # Initialize model and tokenizer
    model, tokenizer = initialize_gpt2(model_name)

    if use_simplified:
        print("Using simplified direct coding approach...")
        solution = simple_gpt2_coding(problem_statement, model, tokenizer)
    else:
        # Create a ToT solver
        tot_solver = GPT2CompetitiveProgrammingToT(
            model=model,
            tokenizer=tokenizer,
            max_depth=3,  # Reduced depth for GPT-2 to avoid drift
            num_thoughts=3,  # Fewer thoughts per level
            evaluation_threshold=0.6  # Lower threshold for GPT-2
        )

        # Solve the problem
        solution = tot_solver.solve(problem_statement)

    # Clean and write the solution
    cpp_code = extract_cpp_code(solution)

    # Create filename if not provided
    if not problem_name:
        # Use a default name
        filename = "solution.cpp"
    else:
        # Clean up the provided name for use as a filename
        problem_name = re.sub(r'[^a-zA-Z0-9_]', '_', problem_name)
        filename = f"{problem_name}.cpp"

    # Write the solution to a file
    cpp_filename = write_cpp_solution(solution, filename)

    return solution, cpp_filename
# Cell 7: Define a simple test problem
SIMPLE_PROBLEM = """
Given two integers, find their sum.

Input:
Two integers a and b (-1000 <= a, b <= 1000).

Output:
The sum of the two integers.

Example:
Input: 3 5
Output: 8
"""

# Cell 8: Run the solver with a simple problem
# Model name can be adjusted based on available resources:
# "gpt2" (smallest, 124M parameters)
# "gpt2-medium" (355M parameters)
# "gpt2-large" (774M parameters)
# "gpt2-xl" (largest, 1.5B parameters)
MODEL_NAME = "gpt2-xl"  # Smaller model for experimentation

# Solve the problem and save the C++ solution
print(f"Solving problem with {MODEL_NAME}...")
solution, cpp_filename = solve_competitive_problem(SIMPLE_PROBLEM, "simple_sum_GPT-2", MODEL_NAME, use_simplified=False)

# Print the solution
print("\nFinal solution:")
print(solution)

# Display the contents of the C++ file
print("\nContents of the C++ file:")
with open(cpp_filename, 'r') as f:
    print(f.read())