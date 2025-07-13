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
# gpt2 Prompt Wrapper
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

class GPT2CompetitiveProgrammingToT:
    def __init__(
        self,
        model=None,
        tokenizer=None,
        model_name="gpt2-xl",
        max_depth=4,
        num_thoughts=3,
        evaluation_threshold=0.6
    ):
        """
        Initialize a Tree of Thoughts solver for competitive programming problems using GPT-2.

        Args:
            model: GPT-2 model (if None, will be initialized)
            tokenizer: GPT-2 tokenizer (if None, will be initialized)
            model_name: Name of the GPT-2 model to use if model/tokenizer not provided
            max_depth: Maximum depth of the tree
            num_thoughts: Number of thoughts to generate at each step
            evaluation_threshold: Threshold for thought evaluation
        """
        # Initialize model and tokenizer if not provided
        if model is None or tokenizer is None:
            self.model, self.tokenizer = initialize_gpt2(model_name)
        else:
            self.model = model
            self.tokenizer = tokenizer

        self.max_depth = max_depth
        self.num_thoughts = num_thoughts
        self.evaluation_threshold = evaluation_threshold
        self.problem_metadata = {}

    def generate_thoughts(self, problem: str, previous_thoughts: List[str] = None) -> List[str]:
        """Generate multiple thoughts for a given problem and previous thoughts."""
        previous_thoughts_text = "\n".join(previous_thoughts) if previous_thoughts else "No previous thoughts yet."

        prompt = f"""
Problem:
{problem}

Previous thoughts:
{previous_thoughts_text}

Generate {self.num_thoughts} different possible next thoughts to solve this problem.
Each thought should represent a step towards solving the problem.
Focus on algorithm selection, data structures, edge cases, and efficiency.

Thought 1:"""

        # Generate thoughts one by one for better control
        thoughts = []
        for i in range(self.num_thoughts):
            current_prompt = prompt
            if i > 0:
                # Add previously generated thoughts to the prompt
                thought_list = "\n".join([f"Thought {j+1}: {thoughts[j]}" for j in range(i)])
                current_prompt = f"{prompt.split('Thought 1:')[0]}{thought_list}\n\nThought {i+1}:"

            # Generate a thought
            generated_texts = generate_text_with_gpt2(
                current_prompt,
                self.model,
                self.tokenizer,
                max_new_tokens=100,  # Shorter length for thoughts
                temperature=0.8  # Slightly higher temperature for diversity
            )

            # Extract a single coherent thought from the generated text
            thought_text = generated_texts[0].strip().split("\n")[0]
            # Remove any "Thought X:" prefixes if they exist
            thought_text = re.sub(r'^Thought \d+:', '', thought_text).strip()

            # Add the thought
            thoughts.append(thought_text)

            # Small delay to avoid overloading
            time.sleep(0.5)

        # Format thoughts with numbering
        formatted_thoughts = [f"{i+1}. {thought}" for i, thought in enumerate(thoughts)]
        return formatted_thoughts

    def evaluate_thoughts(self, problem: str, thoughts: List[str]) -> List[Tuple[str, float]]:
        """Evaluate the quality of each thought for solving the competitive programming problem."""
        # With GPT-2, we'll use a simpler evaluation strategy
        # We'll prompt GPT-2 to rate each thought and then parse the results

        thoughts_text = "\n".join(thoughts)

        prompt = f"""
Problem:
{problem}

Thoughts to evaluate:
{thoughts_text}

Evaluate each thought on a scale from 0.0 to 1.0, where 0.0 means completely irrelevant and 1.0 means directly solves the problem.

Ratings:
Thought 1: """

        # Generate ratings for each thought
        scores = []
        for i in range(len(thoughts)):
            current_prompt = prompt
            if i > 0:
                # Add previously generated ratings to the prompt
                rating_list = "\n".join([f"Thought {j+1}: {scores[j]}" for j in range(i)])
                current_prompt = f"{prompt.split('Thought 1:')[0]}{rating_list}\n\nThought {i+1}: "

            # Generate a rating
            generated_texts = generate_text_with_gpt2(
                current_prompt,
                self.model,
                self.tokenizer,
                max_new_tokens=20,  # Very short for just a number
                temperature=0.3  # Lower temperature for more focused generation
            )

            # Try to extract a numerical rating
            rating_text = generated_texts[0].strip().split("\n")[0]
            # Look for a decimal number
            match = re.search(r'(\d+\.\d+)', rating_text)
            if match:
                score = float(match.group(1))
                # Ensure score is in [0, 1]
                score = max(0.0, min(1.0, score))
            else:
                # Default score if parsing fails
                score = 0.5

            scores.append(score)

            # Small delay to avoid overloading
            time.sleep(0.5)

        return list(zip(thoughts, scores))

    def analyze_problem(self, problem: str) -> Dict[str, Any]:
        """
        Analyze the competitive programming problem to extract key information.

        Args:
            problem: The problem statement

        Returns:
            Dictionary containing metadata about the problem
        """
        prompt = f"""
Problem:
{problem}

Analyze this competitive programming problem and extract key metadata:
- problem_name: A short descriptive name for the problem
- time_complexity: Expected time complexity requirement
- space_complexity: Expected space complexity requirement
- key_concepts: Programming concepts required

Analysis:
problem_name: """

        # Generate problem analysis
        generated_texts = generate_text_with_gpt2(
            prompt,
            self.model,
            self.tokenizer,
            max_new_tokens=300,
            temperature=0.4  # Lower temperature for more focused analysis
        )

        analysis_text = generated_texts[0]

        # Try to extract metadata from the analysis
        metadata = {
            "problem_name": "Competitive Problem",
            "time_complexity": "Unknown",
            "space_complexity": "Unknown",
            "key_concepts": ["Algorithms", "Data Structures"]
        }

        # Extract problem name
        name_match = re.search(r'problem_name:\s*(.*?)(?:$|\n)', analysis_text, re.IGNORECASE)
        if name_match:
            metadata["problem_name"] = name_match.group(1).strip()

        # Extract time complexity
        time_match = re.search(r'time_complexity:\s*(.*?)(?:$|\n)', analysis_text, re.IGNORECASE)
        if time_match:
            metadata["time_complexity"] = time_match.group(1).strip()

        # Extract space complexity
        space_match = re.search(r'space_complexity:\s*(.*?)(?:$|\n)', analysis_text, re.IGNORECASE)
        if space_match:
            metadata["space_complexity"] = space_match.group(1).strip()

        # Extract key concepts
        concepts_match = re.search(r'key_concepts:\s*(.*?)(?:$|\n)', analysis_text, re.IGNORECASE)
        if concepts_match:
            concepts_text = concepts_match.group(1).strip()
            concepts = [concept.strip() for concept in concepts_text.split(',')]
            metadata["key_concepts"] = concepts

        return metadata

    def solve(self, problem: str) -> str:
        """
        Solve the competitive programming problem using Tree of Thoughts approach.

        Args:
            problem: Problem statement to solve

        Returns:
            Final solution as C++ code
        """
        print("Analyzing problem...")
        # First, analyze the problem to get metadata
        self.problem_metadata = self.analyze_problem(problem)
        print(f"Problem analysis: {self.problem_metadata}")

        current_depth = 0
        best_thoughts = []

        print(f"Starting Tree of Thoughts exploration (max depth: {self.max_depth})...")
        while current_depth < self.max_depth:
            print(f"Depth {current_depth + 1}/{self.max_depth}")
            # Generate new thoughts based on current best thoughts
            print("  Generating thoughts...")
            thoughts = self.generate_thoughts(problem, best_thoughts)

            if not thoughts:
                print("  No thoughts generated. Breaking.")
                break

            # Evaluate thoughts
            print("  Evaluating thoughts...")
            evaluated_thoughts = self.evaluate_thoughts(problem, thoughts)

            # Sort thoughts by score
            evaluated_thoughts.sort(key=lambda x: x[1], reverse=True)

            # Display thoughts and their scores
            print("  Thought evaluations:")
            for thought, score in evaluated_thoughts:
                print(f"    - {thought} (Score: {score:.2f})")

            # Keep only thoughts above threshold
            good_thoughts = [t for t, s in evaluated_thoughts if s >= self.evaluation_threshold]

            if not good_thoughts:
                # If no thoughts meet threshold, take the best one
                best_thought = evaluated_thoughts[0][0]
                print(f"  Selected thought: {best_thought} (below threshold but best available)")
                best_thoughts.append(best_thought)
            else:
                best_thought = good_thoughts[0]
                print(f"  Selected thought: {best_thought} (above threshold)")
                best_thoughts.append(best_thought)

            current_depth += 1

        # Generate final solution based on the best thoughts and problem metadata
        print("Generating final solution...")
        final_prompt = f"""
Problem:
{problem}

Analysis and thoughts on the approach:
{" ".join(best_thoughts)}

Write a complete C++ solution to the problem above.
Include proper headers (#include statements) and efficient algorithms.
Handle all edge cases and follow competitive programming best practices.

C++ solution:
#include <iostream>
"""

        # Generate the C++ solution
        generated_texts = generate_text_with_gpt2(
            final_prompt,
            self.model,
            self.tokenizer,
            max_new_tokens=800,  # Longer text for full solution
            temperature=0.6,  # Balance between creativity and focus
            repetition_penalty=1.3  # Higher penalty for solutions
        )

        solution = generated_texts[0]
        print("Solution generated successfully!")

        return solution
