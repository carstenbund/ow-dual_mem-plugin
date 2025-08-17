from __future__ import annotations

class Policy:
    def __init__(
        self,
        link_threshold: float = 0.60,
        dedup_threshold: float = 0.93,
        novelty_min: float = 0.10,
        max_links_per_motif: int = 3,
        symbol_jaccard_cap: float = 0.80,
    ):
        self.link_threshold = link_threshold
        self.dedup_threshold = dedup_threshold
        self.novelty_min = novelty_min
        self.max_links_per_motif = max_links_per_motif
        self.symbol_jaccard_cap = symbol_jaccard_cap

    def should_persist(self, novelty_index: float) -> bool:
        return novelty_index >= self.novelty_min

    def should_link(self, score: float, jaccard: float, links_added: int) -> bool:
        if links_added >= self.max_links_per_motif:
            return False
        if jaccard >= self.symbol_jaccard_cap:
            return False
        # mild brake to avoid cliques
        if links_added >= 2 and score < max(self.link_threshold, 0.70):
            return False
        return score >= self.link_threshold

    def is_duplicate(self, top1_sim: float) -> bool:
        return top1_sim >= self.dedup_threshold
