import os
import re
import random
import pickle
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional


from meta_buffer import MetaBuffer
from common_structures import ThoughtNode
print(os.environ.get("OPENAI_API_KEY"))

# --- Dependency Imports ---
try:
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_openai import ChatOpenAI as RealChatOpenAI

    LANGCHAIN_AVAILABLE = True
except ImportError:
    RealChatOpenAI, SystemMessage, HumanMessage = None, object, object
    LANGCHAIN_AVAILABLE = False



# ==============================================================================
# --- MODULE 1: LLM ABSTRACTION & MOCK ---
# ==============================================================================
class MockChatOpenAI:
    def invoke(self, prompt: Any) -> Any:
        class MockResponseContent:
            def __init__(self, content_text: str): self.content = content_text

        return MockResponseContent("Mocked LLM response. The system is functioning mechanically.")


class Pipeline:
    def __init__(self, model_id: str, api_key: Optional[str] = None):
        self.model_id = model_id
        if api_key and RealChatOpenAI:
            self.llm = RealChatOpenAI(api_key=api_key, model=model_id, temperature=0.7)
        else:
            self.llm = MockChatOpenAI()

    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        if RealChatOpenAI and isinstance(self.llm, RealChatOpenAI):
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            response = self.llm.invoke(messages)
            return getattr(response, 'content', '')
        else:
            # Simplified mock call
            return self.llm.invoke(f"SYSTEM: {system_prompt}\nUSER: {user_prompt}").content






class TDABot:
    """The main orchestrator that runs the entire cognitive pipeline."""

    def __init__(self, model_id: str = "gpt-4o", api_key: Optional[str] = None):
        print("Initializing TDA Bot Orchestrator...")
        self.pipeline = Pipeline(model_id, api_key)
        # CORRECTED: This now correctly uses the imported MetaBuffer class
        # without the 'pipeline' argument, matching your file's definition.
        self.meta_buffer = MetaBuffer()
        self.thought_tree = {}
        self.tda_results = None  # Initialize as None

    def _retrieve_and_instantiate_from_buffer(self, problem: str, top_k: int = 3) -> str:
        """
        Retrieves the most relevant templates from the buffer using an intelligent
        search and then uses the bot's own pipeline to generate a solution.
        """
        # This assumes your imported MetaBuffer has a 'search' method for RAG.
        # If it only has 'get_all_templates', the logic will fall back gracefully.
        if hasattr(self.meta_buffer, 'search'):
            print("Using intelligent RAG search...")
            templates = self.meta_buffer.search(problem, top_k=top_k)
        else:
            print("Falling back to basic template retrieval...")
            templates = self.meta_buffer.get_all_templates()

        if not templates:
            system_prompt = "You are a helpful AI assistant tasked with providing a complete and correct C++ solution."
            user_prompt = problem
        else:
            system_prompt = "You are a master problem solver. Use the provided strategic templates from your knowledge base to construct a complete and correct solution to the user's problem. For coding problems, provide C++ code that fulfills all the restrictions in the problem"
            user_prompt = f"""KNOWLEDGE BASE (Most Relevant Strategies):
{json.dumps(templates, indent=2)}

USER'S PROBLEM:
{problem}

Based on your knowledge base, select the most relevant strategy and use it to provide the best possible final solution."""

        # Use the TDABot's own pipeline instance
        return self.pipeline.get_response(system_prompt, user_prompt)

    def run_full_pipeline(self, problem: str):
        # Dynamically import the modules here to ensure they can be found
        # In a real project, these would be at the top of the file.
        from thought_space import TreeOfThoughtsExplorer
        from tda_analyzer import ThoughtSpaceAnalyzer
        from meta_reasoning_template import MetaReasoner

        print("\n>>> Stage 1: Exploring Solution Space...")
        explorer = TreeOfThoughtsExplorer(llm=self.pipeline, beam_width=2, max_depth=2)
        # Corrected assignment based on your original code
        _, self.thought_tree = explorer.solve(problem)
        if not self.thought_tree:
            print("Pipeline halted: Thought exploration failed.");
            return

        print("\n>>> Stage 2: Analyzing Thought Topology...")
        analyzer = ThoughtSpaceAnalyzer(thought_space=self.thought_tree)
        self.tda_results = analyzer.find_choke_points(dbscan_eps=0.6)

        print("\n>>> Stage 3: Generating Meta-Reasoning Templates...")
        if not self.tda_results:
            new_templates = {}
        else:
            reasoner = MetaReasoner(tda_results=self.tda_results, thought_space=self.thought_tree)
            new_templates = reasoner.generate_templates_from_hubs(num_hubs=2)

        print("\n>>> Stage 4: Updating Knowledge Base...")
        self.meta_buffer.add_templates(new_templates)

        print("\n>>> Stage 5: Synthesizing Final Solution...")
        # This now calls the corrected internal method for RAG
        final_solution = self._retrieve_and_instantiate_from_buffer(problem)

        print("\n\n===================================")
        print("      TDA BOT PIPELINE COMPLETE")
        print("===================================")
        print("\nFinal Informed Solution:")
        print(final_solution)


if __name__ == '__main__':
    # Set the API Key securely. It's better to set this in your system's
    # environment variables rather than hardcoding it.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set.")

    tda_bot = TDABot(api_key=api_key)

    problem_statement = """
       You are an expert competitive programmer. You will be given a problem statement, test case constraints and example test inputs and outputs. Please reason step by step about the solution, then provide a complete implementation in C++17. You should correctly implement the routine(s) described in Implementation Details, without reading or writing anything directly from stdin or to stdout, as input and output are passed through the implemented routines. Assume your code will be run on the OFFICIAL grader, and do not add a main, a sample grader, or any other functionality unless it has been explicitly requested.
Put your final solution within a single code block: ```cpp
<your code here>``` 

# Problem statement (Beech Tree)
Vétyem Woods is a famous woodland with lots of colorful trees.
One of the oldest and tallest beech trees is called Ős Vezér.

The tree Ős Vezér can be modeled as a set of $N$ **nodes** and $N-1$ **edges**.
Nodes are numbered from $0$ to $N-1$ and edges are numbered from $1$ to $N-1$.
Each edge connects two distinct nodes of the tree.
Specifically, edge $i$ ($1 \le i \lt N$) connects node $i$ to node $P[i]$, where $0 \le P[i] \lt i$. Node $P[i]$ is called the **parent** of node $i$, and node $i$ is called a **child** of node $P[i]$.

Each edge has a color. There are $M$ possible edge colors numbered from $1$ to $M$.
The color of edge $i$ is $C[i]$.
Different edges may have the same color.

Note that in the definitions above, the case $i = 0$ does not correspond to an edge of the tree. For convenience, we let $P[0] = -1$ and $C[0] = 0$.

For example, suppose that Ős Vezér has $N = 18$ nodes and $M = 3$ possible edge colors, with $17$ edges described by connections $P = [-1, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 10, 11, 11]$ and colors $C = [0, 1, 2, 3, 1, 2, 3, 1, 3, 3, 2, 1, 1, 2, 2, 1, 2, 3]$. The tree is displayed in the following figure:


Árpád is a talented forester who likes to study specific parts of the tree called **subtrees**.
For each $r$ such that $0 \le r \lt N$, the subtree of node $r$ is the set $T(r)$ of nodes with the following properties:
* Node $r$ belongs to $T(r)$.
* Whenever a node $x$ belongs to $T(r)$, all children of $x$ also belong to $T(r)$.
* No other nodes belong to $T(r)$.

The size of the set $T(r)$ is denoted as $|T(r)|$.

Árpád recently discovered a complicated but interesting subtree property. Árpád's discovery involved a lot of playing with pen and paper, and he suspects you might need to do the same to understand it. He will also show you multiple examples you can then analyze in detail.

Suppose we have a fixed $r$ and a permutation $v_0, v_1, \ldots, v_{|T(r)|-1}$ of the nodes in the subtree $T(r)$.

For each $i$ such that $1 \le i \lt |T(r)|$, let $f(i)$ be the number of times the color $C[v_i]$ appears in the following sequence of $i-1$ colors: $C[v_1], C[v_2], \ldots, C[v_{i-1}]$. 

(Note that $f(1)$ is always $0$ because the sequence of colors in its definition is empty.)

The permutation $v_0, v_1, \ldots, v_{|T(r)|-1}$ is a **beautiful permutation** if and only if all the following properties hold:
* $v_0 = r$.
* For each $i$ such that $1 \le i \lt |T(r)|$, the parent of node $v_i$ is node $v_{f(i)}$.

For any $r$ such that $0 \le r \lt N$, the subtree $T(r)$ is a **beautiful subtree** if and only if there exists a beautiful permutation of the nodes in $T(r)$. Note that according to the definition every subtree which consists of a single node is beautiful.

Consider the example tree above.
It can be shown that the subtrees $T(0)$ and $T(3)$ of this tree are not beautiful.
The subtree $T(14)$ is beautiful, as it consists of a single node.
Below, we will show that the subtree $T(1)$ is also beautiful.

Consider the sequence of distinct integers $[v_0, v_1, v_2, v_3, v_4, v_5, v_6] = [1, 4, 5, 12, 13, 6, 14]$. This sequence is a permutation of the nodes in $T(1)$. The figure below depicts this permutation. The labels attached to the nodes are the indices at which those nodes appear in the permutation.


Clearly, the above sequence is a permutation of the nodes in $T(1)$. We will now verify that it is *beautiful*.

* $v_0 = 1$.
* $f(1) = 0$ since $C[v_1] = C[4] = 1$ appears $0$ times in the sequence $[]$.
  * Correspondingly, the parent of $v_1$ is $v_0$. That is, the parent of node $4$ is node $1$. (Formally, $P[4] = 1$.)
* $f(2) = 0$ since $C[v_2] = C[5] = 2$ appears $0$ times in the sequence $[1]$.
  * Correspondingly, the parent of $v_2$ is $v_0$. That is, the parent of $5$ is $1$.
* $f(3) = 1$ since $C[v_3] = C[12] = 1$ appears $1$ time in the sequence $[1, 2]$.
  * Correspondingly, the parent of $v_3$ is $v_1$. That is, the parent of $12$ is $4$.
* $f(4) = 1$ since $C[v_4] = C[13] = 2$ appears $1$ time in the sequence $[1, 2, 1]$.
  * Correspondingly, the parent of $v_4$ is $v_1$. That is, the parent of $13$ is $4$.
* $f(5) = 0$ since $C[v_5] = C[6] = 3$ appears $0$ times in the sequence $[1, 2, 1, 2]$.
  * Correspondingly, the parent of $v_5$ is $v_0$. That is, the parent of $6$ is $1$.
* $f(6) = 2$ since $C[v_6] = C[14] = 2$ appears $2$ times in the sequence $[1, 2, 1, 2, 3]$.
  * Correspondingly, the parent of $v_6$ is $v_2$. That is, the parent of $14$ is $5$.

As we could find a beautiful permutation of the nodes in $T(1)$, the subtree $T(1)$ is indeed beautiful.

Your task is to help Árpád decide for every subtree of Ős Vezér whether it is beautiful.

## Implementation Details

You should implement the following procedure.

```
int[] beechtree(int N, int M, int[] P, int[] C)
```

* $N$: the number of nodes in the tree.
* $M$: the number of possible edge colors.
* $P$, $C$: arrays of length $N$ describing the edges of the tree.
* This procedure should return an array $b$ of length $N$. 
  For each $r$ such that $0 \le r \lt N$, $b[r]$ should be $1$ if $T(r)$ is beautiful, and $0$ otherwise.
* This procedure is called exactly once for each test case.

## Examples

### Example 1

Consider the following call:

```
beechtree(4, 2, [-1, 0, 0, 0], [0, 1, 1, 2])
```

The tree is displayed in the following figure:


$T(1)$, $T(2)$, and $T(3)$ each consist of a single node and are therefore beautiful.
$T(0)$ is not beautiful.
Therefore, the procedure should return $[0, 1, 1, 1]$.

### Example 2

Consider the following call:

```
beechtree(18, 3, 
          [-1, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 10, 11, 11],
          [0, 1, 2, 3, 1, 2, 3, 1, 3, 3, 2, 1, 1, 2, 2, 1, 2, 3])
```

This example is illustrated in the task description above.

The procedure should return $[0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]$.

### Example 3

Consider the following call:

```
beechtree(7, 2, [-1, 0, 1, 1, 0, 4, 5], [0, 1, 1, 2, 2, 1, 1])
```

This example is illustrated in the following figure.


$T(0)$ is the only subtree that is not beautiful.
The procedure should return $[0, 1, 1, 1, 1, 1, 1]$.


## Constraints
* $3 \le N \le 200\,000$
* $2 \le M \le 200\,000$
* $0 \le P[i] \lt i$ (for each $i$ such that $1 \le i \lt N$)
* $1 \le C[i] \le M$ (for each $i$ such that $1 \le i \lt N$)
* $P[0] = -1$ and $C[0] = 0$


## Sample Grader

The sample grader reads the input in the following format:

* line $1$: $N \; M$
* line $2$: $P[0] \; P[1] \; \ldots \; P[N-1]$
* line $3$: $C[0] \; C[1] \; \ldots \; C[N-1]$

Let $b[0], \; b[1], \; \ldots$ denote the elements of the array returned by `beechtree`.
The sample grader prints your answer in a single line, in the following format:
* line $1$: $b[0] \; b[1] \; \ldots$

## Execution limits
Your solution will have 1.5 second(s) execution time and 2048 MB memory limit to solve each test case.

## Starting code
Here's your starting code with some skeleton/placeholder functionality:
```cpp
#include "beechtree.h"

std::vector<int> beechtree(int N, int M, std::vector<int> P, std::vector<int> C)
{
    return {};
}

       """

    tda_bot.run_full_pipeline(problem=problem_statement)
