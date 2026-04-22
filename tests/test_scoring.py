from storygen.scoring.clip_consistency import CLIPConsistencyScorer
from storygen.types import CandidateScore, GenerationCandidate, PromptSpec, Scene, SceneSelectionResult, Story


class FakeCLIPConsistencyScorer(CLIPConsistencyScorer):
    def __init__(self) -> None:
        super().__init__(
            {
                "type": "clip_consistency",
                "clip_model_id": "fake",
                "text_image_weight": 0.5,
                "consistency_weight": 0.3,
                "action_weight": 0.2,
            },
            {"device": "cpu"},
        )
        self.text_scores = {
            ("cand_a.png", "full"): 0.8,
            ("cand_a.png", "local"): 0.6,
            ("cand_b.png", "full"): 0.4,
            ("cand_b.png", "local"): 0.9,
        }
        self.consistency_scores = {
            ("cand_a.png", "prev.png"): 0.7,
            ("cand_b.png", "prev.png"): 0.2,
        }

    def _text_alignment_score(self, image_path: str | None, text: str) -> float:
        if not image_path:
            return 0.0
        key = "local" if "action" in text else "full"
        return self.text_scores[(image_path, key)]

    def _image_consistency_score(self, image_path: str | None, previous_image_path: str | None) -> float:
        if not image_path or not previous_image_path:
            return 0.0
        return self.consistency_scores[(image_path, previous_image_path)]


def _scene() -> Scene:
    return Scene("SCENE-1", 0, "raw", "clean", ["Lily"])


def _story() -> Story:
    return Story("story.txt", "", [_scene()], ["Lily"], ["Lily"], {"Lily": ["SCENE-1"]})


def _prompt_spec() -> PromptSpec:
    return PromptSpec(
        scene_id="SCENE-1",
        style_prompt="style",
        character_prompt="character",
        global_context_prompt="context",
        local_prompt="action prompt",
        action_prompt="short action prompt",
        generation_prompt="short generation prompt",
        scoring_prompt="short scoring prompt",
        full_prompt="full prompt",
        negative_prompt="negative",
    )


def _candidate(index: int, image_path: str, metadata: dict | None = None) -> GenerationCandidate:
    return GenerationCandidate(
        scene_id="SCENE-1",
        candidate_index=index,
        seed=100 + index,
        prompt_spec=_prompt_spec(),
        image=None,
        image_path=image_path,
        metadata=metadata or {},
    )


def test_clip_scorer_omits_previous_image_weight_for_first_scene() -> None:
    scorer = FakeCLIPConsistencyScorer()
    candidate = _candidate(0, "cand_a.png")

    score = scorer.score_candidate(
        story=_story(),
        scene=_scene(),
        prompt_spec=_prompt_spec(),
        candidate=candidate,
        previous_results=[],
    )

    assert score.components["text_alignment"] == 0.8
    assert score.components["action_alignment"] == 0.6
    assert score.components["previous_image_consistency"] == 0.0
    assert score.score == 0.742857


def test_clip_scorer_includes_previous_image_consistency_for_later_scenes() -> None:
    scorer = FakeCLIPConsistencyScorer()
    candidate = _candidate(0, "cand_a.png")
    previous_result = SceneSelectionResult(
        scene_id="SCENE-0",
        selected_candidate_index=0,
        selected_seed=1,
        selected_image_path="prev.png",
        selected_score=CandidateScore("SCENE-0", 0, 1, 1.0, "fake"),
        candidate_scores=[],
        candidate_image_paths=["prev.png"],
    )

    score = scorer.score_candidate(
        story=_story(),
        scene=_scene(),
        prompt_spec=_prompt_spec(),
        candidate=candidate,
        previous_results=[previous_result],
    )

    assert score.components["previous_image_consistency"] == 0.7
    assert score.score == 0.73


def test_clip_scorer_selects_highest_weighted_candidate() -> None:
    scorer = FakeCLIPConsistencyScorer()
    previous_result = SceneSelectionResult(
        scene_id="SCENE-0",
        selected_candidate_index=0,
        selected_seed=1,
        selected_image_path="prev.png",
        selected_score=CandidateScore("SCENE-0", 0, 1, 1.0, "fake"),
        candidate_scores=[],
        candidate_image_paths=["prev.png"],
    )
    candidate_a = _candidate(0, "cand_a.png")
    candidate_b = _candidate(1, "cand_b.png")
    score_a = scorer.score_candidate(
        story=_story(),
        scene=_scene(),
        prompt_spec=_prompt_spec(),
        candidate=candidate_a,
        previous_results=[previous_result],
    )
    score_b = scorer.score_candidate(
        story=_story(),
        scene=_scene(),
        prompt_spec=_prompt_spec(),
        candidate=candidate_b,
        previous_results=[previous_result],
    )

    selection = scorer.select_best(
        scene_id="SCENE-1",
        candidates=[candidate_a, candidate_b],
        candidate_scores=[score_a, score_b],
    )

    assert selection.selected_candidate_index == 0


def test_clip_scorer_prefers_short_fields_and_falls_back_only_when_missing() -> None:
    scorer = FakeCLIPConsistencyScorer()
    prompt_spec = _prompt_spec()

    assert scorer._build_scoring_text(prompt_spec) == "short scoring prompt"
    assert scorer._build_action_text(prompt_spec) == "short action prompt"

    fallback_prompt_spec = PromptSpec(
        scene_id="SCENE-1",
        style_prompt="style",
        character_prompt="character",
        global_context_prompt="context",
        local_prompt="local fallback",
        action_prompt="",
        generation_prompt="generation fallback",
        scoring_prompt="",
        full_prompt="full fallback",
        negative_prompt="negative",
    )

    assert scorer._build_scoring_text(fallback_prompt_spec) == "generation fallback"
    assert scorer._build_action_text(fallback_prompt_spec) == "local fallback"


def test_route_aware_scoring_disabled_preserves_existing_calculation() -> None:
    scorer = FakeCLIPConsistencyScorer()
    candidate = _candidate(0, "cand_a.png", {"route_change_level": "medium"})
    previous_result = SceneSelectionResult(
        scene_id="SCENE-0",
        selected_candidate_index=0,
        selected_seed=1,
        selected_image_path="prev.png",
        selected_score=CandidateScore("SCENE-0", 0, 1, 1.0, "fake"),
        candidate_scores=[],
        candidate_image_paths=["prev.png"],
    )

    score = scorer.score_candidate(
        story=_story(),
        scene=_scene(),
        prompt_spec=_prompt_spec(),
        candidate=candidate,
        previous_results=[previous_result],
    )

    assert score.score == 0.73
    assert score.components["over_similarity_penalty"] == 0.0
    assert score.metadata["route_aware"] == {"enabled": False}


def test_route_aware_medium_lowers_consistency_weight_and_applies_penalty() -> None:
    scorer = FakeCLIPConsistencyScorer()
    scorer.scoring_config["route_aware"] = {
        "enabled": True,
        "consistency_weight_by_change_level": {"small": 0.3, "medium": 0.1, "large": 0.0},
        "over_similarity_penalty": {
            "enabled": True,
            "threshold_by_change_level": {"small": None, "medium": 0.6, "large": 0.5},
            "penalty_weight_by_change_level": {"medium": 0.2, "large": 0.25},
        },
    }
    candidate = _candidate(0, "cand_a.png", {"route_change_level": "medium"})
    previous_result = SceneSelectionResult(
        scene_id="SCENE-0",
        selected_candidate_index=0,
        selected_seed=1,
        selected_image_path="prev.png",
        selected_score=CandidateScore("SCENE-0", 0, 1, 1.0, "fake"),
        candidate_scores=[],
        candidate_image_paths=["prev.png"],
    )

    score = scorer.score_candidate(
        story=_story(),
        scene=_scene(),
        prompt_spec=_prompt_spec(),
        candidate=candidate,
        previous_results=[previous_result],
    )

    assert score.metadata["weights"]["previous_image_consistency"] == 0.1
    assert score.metadata["raw_weighted_score"] == 0.7375
    assert score.components["over_similarity_penalty"] == 0.05
    assert score.score == 0.6875


def test_route_aware_large_can_select_less_similar_more_semantic_candidate() -> None:
    scorer = FakeCLIPConsistencyScorer()
    scorer.scoring_config["route_aware"] = {
        "enabled": True,
        "consistency_weight_by_change_level": {"small": 0.3, "medium": 0.1, "large": 0.0},
        "over_similarity_penalty": {
            "enabled": True,
            "threshold_by_change_level": {"small": None, "medium": 0.9, "large": 0.8},
            "penalty_weight_by_change_level": {"medium": 0.15, "large": 0.3},
        },
    }
    scorer.text_scores.update(
        {
            ("cand_a.png", "full"): 0.55,
            ("cand_a.png", "local"): 0.55,
            ("cand_b.png", "full"): 0.85,
            ("cand_b.png", "local"): 0.85,
        }
    )
    scorer.consistency_scores.update(
        {
            ("cand_a.png", "prev.png"): 0.98,
            ("cand_b.png", "prev.png"): 0.7,
        }
    )
    previous_result = SceneSelectionResult(
        scene_id="SCENE-0",
        selected_candidate_index=0,
        selected_seed=1,
        selected_image_path="prev.png",
        selected_score=CandidateScore("SCENE-0", 0, 1, 1.0, "fake"),
        candidate_scores=[],
        candidate_image_paths=["prev.png"],
    )
    candidate_a = _candidate(0, "cand_a.png", {"route_change_level": "large"})
    candidate_b = _candidate(1, "cand_b.png", {"route_change_level": "large"})

    score_a = scorer.score_candidate(
        story=_story(),
        scene=_scene(),
        prompt_spec=_prompt_spec(),
        candidate=candidate_a,
        previous_results=[previous_result],
    )
    score_b = scorer.score_candidate(
        story=_story(),
        scene=_scene(),
        prompt_spec=_prompt_spec(),
        candidate=candidate_b,
        previous_results=[previous_result],
    )
    selection = scorer.select_best(
        scene_id="SCENE-1",
        candidates=[candidate_a, candidate_b],
        candidate_scores=[score_a, score_b],
    )

    assert score_a.components["over_similarity_penalty"] == 0.27
    assert score_b.components["over_similarity_penalty"] == 0.0
    assert selection.selected_candidate_index == 1
