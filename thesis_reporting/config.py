"""Configuration for result-addition scripts.

The definitions keep analyses tied to explicit local result
artifacts instead of scanning the full result tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResultSystem:
    """Describe one final evaluation condition used in result additions."""

    key: str
    label: str
    phase: str
    split: str
    architecture: str
    representation: str
    scope: str
    condition_id: str
    condition_dir: Path

    @property
    def per_author_metrics_path(self) -> Path:
        """Return the condition-level per-author metrics path."""

        return self.condition_dir / "diagnostics/per_author_metrics.csv"

    @property
    def final_predictions_path(self) -> Path:
        """Return the condition-level final prediction path."""

        return self.condition_dir / "final_test_predictions.csv"

    @property
    def confusion_pairs_path(self) -> Path:
        """Return the condition-level directed confusion-pairs path."""

        return self.condition_dir / "diagnostics/confusion_pairs.csv"

    @property
    def top_confusions_path(self) -> Path:
        """Return the condition-level top-confusions path."""

        return self.condition_dir / "diagnostics/top_confusions.csv"

    @property
    def normalized_confusion_matrix_path(self) -> Path:
        """Return the condition-level normalized confusion matrix path."""

        return self.condition_dir / "diagnostics/normalized_confusion_matrix.csv"

    @property
    def normalized_confusion_heatmap_path(self) -> Path:
        """Return the condition-level normalized confusion heatmap path."""

        return (
            self.condition_dir / "diagnostics/figures/normalized_confusion_heatmap.png"
        )


@dataclass(frozen=True)
class SystemComparison:
    """Describe one pairwise result-addition comparison."""

    key: str
    label: str
    source_system_key: str
    target_system_key: str
    purpose: str
    comparison_group: str


@dataclass(frozen=True)
class ProfileTarget:
    """Describe one profiling target used by attribution profile-quality outputs."""

    key: str
    label: str
    prediction_file_key: str


@dataclass(frozen=True)
class ProfileQualityRun:
    """Describe the configured profile-quality artifact bundle."""

    key: str
    label: str
    quality_dir: Path

    @property
    def attribution_test_metrics_path(self) -> Path:
        """Return profile metrics for the attribution final-test speeches."""

        return self.quality_dir / "attribution_test_profile_metrics.csv"

    @property
    def calibration_summary_path(self) -> Path:
        """Return profile calibration summary path."""

        return self.quality_dir / "calibration_summary.csv"

    @property
    def target_summary_path(self) -> Path:
        """Return profile target-summary path."""

        return self.quality_dir / "target_summary.csv"

    def prediction_path(self, target: ProfileTarget, role: str) -> Path:
        """Return profile predictions for one attribution split role and target."""

        return (
            self.quality_dir
            / "predictions"
            / (
                "attribution_final_test_"
                f"{role}_{target.prediction_file_key}_profile_predictions.csv"
            )
        )


def configured_systems() -> tuple[ResultSystem, ...]:
    """Return the explicit final systems used by result-addition scripts."""

    return (
        ResultSystem(
            key="phase1a_char_word",
            label="Phase 1A direct lexical baseline",
            phase="phase1a",
            split="bokmal_authorwise",
            architecture="direct",
            representation="none",
            scope="baseline",
            condition_id="char_word",
            condition_dir=Path(
                "models/bokmal_authorwise/bokmal_authorwise_final_linear_svm/"
                "seed_42/final_by_condition/char_word"
            ),
        ),
        ResultSystem(
            key="phase1b_char_word",
            label="Phase 1B stacked lexical baseline",
            phase="phase1b",
            split="bokmal_authorwise",
            architecture="stacked",
            representation="none",
            scope="baseline",
            condition_id="char_word",
            condition_dir=Path(
                "models/bokmal_authorwise/bokmal_authorwise_final_stacked/"
                "seed_42/final_by_condition/char_word"
            ),
        ),
        ResultSystem(
            key="phase3a_probability_all",
            label="Phase 3A predicted probability all-signal",
            phase="phase3a",
            split="bokmal_authorwise",
            architecture="direct",
            representation="probability",
            scope="all_signal",
            condition_id="char_word_profiling_all",
            condition_dir=Path(
                "models/bokmal_authorwise/"
                "bokmal_authorwise_final_linear_svm_with_profiling/"
                "seed_42/final_by_condition/char_word_profiling_all"
            ),
        ),
        ResultSystem(
            key="phase3b_probability_all",
            label="Phase 3B predicted probability all-signal",
            phase="phase3b",
            split="bokmal_authorwise",
            architecture="stacked",
            representation="probability",
            scope="all_signal",
            condition_id="char_word__profiling_all",
            condition_dir=Path(
                "models/bokmal_authorwise/"
                "bokmal_authorwise_final_stacked_with_profiling/"
                "seed_42/final_by_condition/char_word__profiling_all"
            ),
        ),
        ResultSystem(
            key="phase3a_oracle_all",
            label="Phase 3A oracle all-signal",
            phase="phase3a",
            split="bokmal_authorwise",
            architecture="direct",
            representation="oracle",
            scope="all_signal",
            condition_id="char_word_stylo_downweighted_oracle_profiling_all",
            condition_dir=Path(
                "models/bokmal_authorwise/"
                "bokmal_authorwise_final_linear_svm_with_oracle_profiling/"
                "seed_42/final_by_condition/"
                "char_word_stylo_downweighted_oracle_profiling_all"
            ),
        ),
        ResultSystem(
            key="phase3b_oracle_all",
            label="Phase 3B oracle all-signal",
            phase="phase3b",
            split="bokmal_authorwise",
            architecture="stacked",
            representation="oracle",
            scope="all_signal",
            condition_id="char_word_stylo__oracle_all",
            condition_dir=Path(
                "models/bokmal_authorwise/"
                "bokmal_authorwise_final_stacked_with_oracle_profiling/"
                "seed_42/final_by_condition/char_word_stylo__oracle_all"
            ),
        ),
        ResultSystem(
            key="temporal_phase1a_char_word",
            label="Phase 1A temporal direct lexical baseline",
            phase="phase1a_temporal",
            split="bokmal_temporal",
            architecture="direct",
            representation="none",
            scope="temporal",
            condition_id="char_word",
            condition_dir=Path(
                "models/bokmal_temporal/bokmal_temporal_final_linear_svm/"
                "seed_42/final_by_condition/char_word"
            ),
        ),
    )


def configured_feature_importance_systems() -> tuple[ResultSystem, ...]:
    """Return final systems whose saved models support importance analysis."""

    return tuple(
        system for system in configured_systems() if system.scope != "temporal"
    )


def configured_profile_targets() -> tuple[ProfileTarget, ...]:
    """Return profile targets used by profile-quality result additions."""

    return (
        ProfileTarget(
            key="party",
            label="Party",
            prediction_file_key="party",
        ),
        ProfileTarget(
            key="female",
            label="Gender",
            prediction_file_key="female",
        ),
        ProfileTarget(
            key="age_bin",
            label="Age bin",
            prediction_file_key="age_bin",
        ),
        ProfileTarget(
            key="left_center_right",
            label="Left-center-right bloc",
            prediction_file_key="left_center_right",
        ),
    )


def configured_profile_quality_run() -> ProfileQualityRun:
    """Return the profile-quality artifact bundle used by result additions."""

    return ProfileQualityRun(
        key="bokmal_authorwise_profile_quality",
        label="Bokmal authorwise profile-quality diagnostics",
        quality_dir=Path(
            "profiling_quality/bokmal_authorwise/" "bokmal_profiling_linear_svm/seed_42"
        ),
    )


def configured_comparisons() -> tuple[SystemComparison, ...]:
    """Return explicit pairwise comparisons used by result-addition scripts."""

    return (
        SystemComparison(
            key="phase1a_vs_phase3a_probability_all",
            label="Phase 1A baseline to Phase 3A predicted all-signal",
            source_system_key="phase1a_char_word",
            target_system_key="phase3a_probability_all",
            purpose="Direct architecture predicted-profile effect",
            comparison_group="predicted_profile_effect",
        ),
        SystemComparison(
            key="phase1a_vs_phase3a_oracle_all",
            label="Phase 1A baseline to Phase 3A oracle all-signal",
            source_system_key="phase1a_char_word",
            target_system_key="phase3a_oracle_all",
            purpose="Direct architecture oracle-profile effect",
            comparison_group="oracle_profile_effect",
        ),
        SystemComparison(
            key="phase3a_probability_vs_oracle_all",
            label="Phase 3A predicted all-signal to Phase 3A oracle all-signal",
            source_system_key="phase3a_probability_all",
            target_system_key="phase3a_oracle_all",
            purpose="Direct architecture profile-quality bottleneck",
            comparison_group="oracle_predicted_gap",
        ),
        SystemComparison(
            key="phase1b_vs_phase3b_probability_all",
            label="Phase 1B baseline to Phase 3B predicted all-signal",
            source_system_key="phase1b_char_word",
            target_system_key="phase3b_probability_all",
            purpose="Stacked architecture predicted-profile effect",
            comparison_group="predicted_profile_effect",
        ),
        SystemComparison(
            key="phase1b_vs_phase3b_oracle_all",
            label="Phase 1B baseline to Phase 3B oracle all-signal",
            source_system_key="phase1b_char_word",
            target_system_key="phase3b_oracle_all",
            purpose="Stacked architecture oracle-profile effect",
            comparison_group="oracle_profile_effect",
        ),
        SystemComparison(
            key="phase3b_probability_vs_oracle_all",
            label="Phase 3B predicted all-signal to Phase 3B oracle all-signal",
            source_system_key="phase3b_probability_all",
            target_system_key="phase3b_oracle_all",
            purpose="Stacked architecture profile-quality bottleneck",
            comparison_group="oracle_predicted_gap",
        ),
        SystemComparison(
            key="phase1a_vs_phase1b",
            label="Phase 1A direct baseline to Phase 1B stacked baseline",
            source_system_key="phase1a_char_word",
            target_system_key="phase1b_char_word",
            purpose="Direct versus stacked lexical baseline",
            comparison_group="architecture_effect",
        ),
        SystemComparison(
            key="phase3a_probability_vs_phase3b_probability_all",
            label="Phase 3A predicted all-signal to Phase 3B predicted all-signal",
            source_system_key="phase3a_probability_all",
            target_system_key="phase3b_probability_all",
            purpose="Direct versus stacked predicted-profile system",
            comparison_group="architecture_effect",
        ),
        SystemComparison(
            key="phase3a_oracle_vs_phase3b_oracle_all",
            label="Phase 3A oracle all-signal to Phase 3B oracle all-signal",
            source_system_key="phase3a_oracle_all",
            target_system_key="phase3b_oracle_all",
            purpose="Direct versus stacked oracle-profile system",
            comparison_group="architecture_effect",
        ),
    )
