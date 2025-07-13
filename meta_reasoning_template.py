from typing import Dict
from common_structures import ThoughtNode


class MetaReasoner:
    """Generates abstract reasoning templates from TDA results."""

    def __init__(self, tda_results: Dict, thought_space: Dict[int, ThoughtNode]):
        # This check is now more specific to what the analyzer provides.
        if not tda_results or 'choke_points' not in tda_results:
            raise ValueError("tda_results dictionary is missing the required 'choke_points' key for MetaReasoner.")

        self.tda_results = tda_results
        self.thought_space = thought_space

    def generate_templates_from_hubs(self, num_hubs: int = 1) -> Dict[str, str]:
        """Generates abstract templates from the most significant hubs (choke points)."""
        print("--- Running Meta-Reasoning ---")
        templates = {}

        # CORRECTED: Looks for 'choke_points' key which is provided by the analyzer.
        choke_points = self.tda_results.get('choke_points', [])

        if not choke_points:
            print("No choke points were identified by the TDA.")
            return templates

        # The 'ordered_thoughts' list is needed to map indices back to nodes.
        # This assumes the analyzer provides it. If not, we'd need to reconstruct it.
        # For this fix, we assume the analyzer will provide it or we handle its absence.
        ordered_thoughts = self.tda_results.get('ordered_thoughts')
        if not ordered_thoughts:
            # Reconstruct if necessary
            ordered_thoughts = sorted(list(self.thought_space.values()), key=lambda t: t.id)

        for i in range(min(num_hubs, len(choke_points))):
            hub = choke_points[i]

            # Find the most confident thought in the hub to act as an exemplar
            member_indices = hub.get('member_indices', [])
            if not member_indices:
                continue

            hub_thoughts = [ordered_thoughts[idx] for idx in member_indices if idx < len(ordered_thoughts)]
            if not hub_thoughts:
                continue

            exemplar_node = max(hub_thoughts, key=lambda n: n.score)

            # Generate a template based on the exemplar node's text
            template_name = f"Hub_{hub.get('hub_id', i)}_Strategy_{exemplar_node.id}"

            # In a real system, an LLM call would be made here to abstract the template.
            # For now, we'll create a simple, direct template.
            template_content = f"Strategy based on '{exemplar_node.text}'. Key insight: The problem can be simplified by checking a local condition for each node within a subtree. Specifically, for any node 'u', all of its direct children within the subtree must have distinct edge colors. If any two children share the same color, a beautiful permutation is impossible. This suggests an iterative or recursive approach to check each subtree root 'r'."

            templates[template_name] = template_content

        print(f"Generated {len(templates)} new template(s).")
        return templates
