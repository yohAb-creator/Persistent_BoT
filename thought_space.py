import os
import re
import random
from typing import List, Tuple, Dict, Any, Optional

# This assumes you have these files in the same directory
from common_structures import ThoughtNode


# The direct import of Pipeline is removed, as it's defined in the main script.

class TreeOfThoughtsExplorer:
    """ An implementation of ToT to acquire thought trajectories """

    # The type hint for 'llm' is changed to a string forward reference ('Pipeline')
    # to avoid a circular import issue, as Pipeline is defined in the main script.
    def __init__(self, llm: 'Pipeline', beam_width=3, max_depth=4, num_children_per_node=3):
        self.llm = llm
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.num_children_per_node = num_children_per_node
        self.nodes: Dict[int, ThoughtNode] = {}
        self.next_id: int = 0

    def _get_new_id(self) -> int:
        new_id = self.next_id
        self.next_id += 1
        return new_id

    def solve(self, problem: str) -> Tuple[Optional[str], Dict[int, ThoughtNode]]:
        self.nodes.clear()
        self.next_id = 0
        promising_nodes = self._generate_initial_thoughts(problem)
        beam = promising_nodes[:self.beam_width]

        for depth in range(self.max_depth):
            print(f"\n--- Exploring Depth {depth + 1} (Beam Width: {len(beam)}) ---")
            if not beam:
                print("No more promising thoughts to expand. Halting search.")
                break
            all_children_this_level = []
            for parent_node in beam:
                children = self._expand_node(parent_node, problem)
                all_children_this_level.extend(children)

            all_children_this_level.sort(key=lambda n: n.score, reverse=True)
            beam = [child for child in all_children_this_level if child.is_promising][:self.beam_width]

        if not self.nodes:
            return "No solution could be generated.", {}

        best_leaf = max([n for n in self.nodes.values() if not n.children_ids], key=lambda n: n.score,
                        default=max(self.nodes.values(), key=lambda n: n.score))

        path = []
        curr = best_leaf
        while curr:
            path.append(curr.text)
            curr = self.nodes.get(curr.parent_id)
        path.reverse()

        system_prompt = "You are an expert programmer. Based on the provided reasoning path, generate the final C++ solution."
        user_prompt = f"Problem: {problem}\nBased on this reasoning: {' -> '.join(path)}\nProvide the final solution."
        final_solution = self.llm.get_response(system_prompt, user_prompt)

        return final_solution, self.nodes

    def _generate_initial_thoughts(self, problem: str) -> List[ThoughtNode]:
        """Correctly implemented generation method."""
        print("\n--- Generating Initial Thoughts ---")
        system_prompt = "You are an IOI Gold Medalist and legendary competitive programming coach..."
        user_prompt = f"Problem Statement:\n{problem}\n\nGenerate 5 diverse initial strategies:"

        # CORRECTED: Passing two arguments as required by Pipeline.get_response
        response_content = self.llm.get_response(system_prompt, user_prompt)
        thought_texts = response_content.strip().split('\n')

        initial_nodes = []
        for text in thought_texts:
            if not text.strip(): continue
            node = ThoughtNode(id=self._get_new_id(), text=text, depth=0)

            paradigm_system_prompt = "Classify the following thought into one of these paradigms [DP, Greedy, Graph Algorithm, Other]. Respond with only the paradigm name."
            paradigm_user_prompt = f"Thought: {text}"
            # CORRECTED: Passing two arguments
            node.paradigm = self.llm.get_response(paradigm_system_prompt, paradigm_user_prompt).strip()

            self.nodes[node.id] = node
            initial_nodes.append(node)

        print(f"Generated {len(initial_nodes)} initial thoughts.")
        return self._evaluate_and_score(initial_nodes, problem)

    def _expand_node(self, parent_node: ThoughtNode, problem: str) -> List[ThoughtNode]:
        """Correctly implemented expansion method."""
        if not parent_node.is_promising: return []
        print(f"  Expanding Node ID {parent_node.id} (Depth {parent_node.depth})")
        system_prompt = "You are a focused competitive programmer translating a high-level strategy into a concrete plan..."
        user_prompt = f"Problem: {problem}\nHigh-Level Strategy: {parent_node.text}\n\nElaborate with {self.num_children_per_node} concrete implementation steps:"

        # CORRECTED: Passing two arguments
        response_content = self.llm.get_response(system_prompt, user_prompt)
        child_texts = response_content.strip().split('\n')

        children = []
        for text in child_texts:
            if not text.strip(): continue
            child_node = ThoughtNode(id=self._get_new_id(), text=text, depth=parent_node.depth + 1,
                                     parent_id=parent_node.id, paradigm=parent_node.paradigm)
            self.nodes[child_node.id] = child_node
            parent_node.children_ids.append(child_node.id)
            children.append(child_node)
        return self._evaluate_and_score(children, problem)

    def _evaluate_and_score(self, nodes: List[ThoughtNode], problem: str) -> List[ThoughtNode]:
        if not nodes: return []
        print(f"  Evaluating {len(nodes)} thoughts...")
        system_prompt = "You are a skeptical but fair judge at a competitive programming contest. Provide only a numerical score between 0.0 and 1.0 for each thought, one per line. Do not add any other text."
        thoughts_for_prompt = "\n".join([f"{i + 1}. Thought: {n.text}" for i, n in enumerate(nodes)])
        user_prompt = f"Problem:\n{problem}\n\nEvaluate these thoughts:\n{thoughts_for_prompt}"

        # CORRECTED: Passing two arguments
        response_content = self.llm.get_response(system_prompt, user_prompt)
        scores_text = response_content.strip().split('\n')

        scores = []
        for line in scores_text:
            match = re.search(r'(\d\.\d+)', line)
            if not match: match = re.search(r'(\d(?:\.\d*)?)', line)
            if match:
                try:
                    scores.append(max(0.0, min(1.0, float(match.group(1)))))
                except (ValueError, IndexError):
                    scores.append(0.0)
            else:
                scores.append(0.0)

        while len(scores) < len(nodes): scores.append(0.0)

        for i, node in enumerate(nodes):
            node.score = scores[i]
            if node.score < 0.3: node.is_promising = False

        nodes.sort(key=lambda n: n.score, reverse=True)
        return nodes
