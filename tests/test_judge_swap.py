from pathlib import Path

from experiments.analysis.judge_swap import _label


def test_label_reads_stamped_paper_directory():
    d = Path("data/results/paper/20260324_153827_rq1_musique_n200_gemini-3.1-flash-lite")
    assert _label(d) == "musique / gemini-3.1-flash-lite"


def test_label_reads_two_part_dataset_and_model():
    d = Path("20260324_161133_rq1_2wikimultihopqa_n200_gpt-5-mini")
    assert _label(d) == "2wikimultihopqa / gpt-5-mini"


def test_label_falls_back_to_gpt_oss_for_openweight_directory():
    assert _label(Path("data/results-openweight/final/hotpotqa")) == "hotpotqa / gpt-oss"
