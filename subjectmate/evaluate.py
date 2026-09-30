import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from subjectmate.settings import Settings
from subjectmate.ranking import is_course_material
from subjectmate.vectorstore import index_is_current, load_index, retrieve


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Evaluation file must contain a JSON array")
    required = {"id", "question", "expected_outcome", "supporting_sources"}
    for case in cases:
        missing = required - set(case)
        if missing:
            raise ValueError(f"Evaluation case missing fields: {sorted(missing)}")
        if case["expected_outcome"] not in {"answer", "partial", "abstain"}:
            raise ValueError(f"Invalid expected_outcome in {case['id']}")
    return cases


def _pages(metadata: dict[str, Any]) -> list[int]:
    """PDF pages covered by a chunk; merged sections span a range such as "4-6"."""
    if metadata.get("location_type") != "page":
        return []
    first, _, last = str(metadata.get("location", "")).partition("-")
    return list(range(int(first), int(last or first) + 1))


def matches_source(metadata: dict[str, Any], sources: list[dict[str, Any]]) -> bool:
    """A chunk supports a case if it comes from a listed file (and page, when pages are given)."""
    for source in sources:
        if metadata.get("source") != source["filename"]:
            continue
        pages = source.get("pages") or []
        if not pages or set(pages) & set(_pages(metadata)):
            return True
    return False


def evaluate_retrieval(settings: Settings, k: int) -> dict[str, Any]:
    if not index_is_current(settings):
        raise ValueError("Build a current index before running evaluation")
    cases = load_cases(settings.evaluation_file)
    if not cases:
        raise ValueError(f"Add benchmark cases to {settings.evaluation_file}")

    vectorstore = load_index(settings)
    retrieval_settings = replace(
        settings, retrieval_k=k, retrieval_fetch_k=max(k, settings.retrieval_fetch_k)
    )
    retrieve("warm-up", vectorstore, retrieval_settings)  # load models before timing
    rows = []
    answerable = abstain = hits = course_top = wrongly_blocked = off_topic_blocked = 0
    reciprocal_ranks: list[float] = []
    seconds: list[float] = []
    for case in cases:
        started = time.perf_counter()
        retrieved = retrieve(case["question"], vectorstore, retrieval_settings)
        seconds.append(time.perf_counter() - started)
        results = [
            {
                "filename": document.metadata.get("source"),
                "location": document.metadata.get("location"),
                "course": is_course_material(document.metadata),
                "supports": matches_source(document.metadata, case["supporting_sources"]),
                "score": round(float(score), 4),
                "text": document.page_content[:300],
            }
            for document, score in retrieved
        ]
        rank = next((index for index, item in enumerate(results, 1) if item["supports"]), None)
        if case["expected_outcome"] == "abstain":
            abstain += 1
            off_topic_blocked += int(not results)
        else:
            answerable += 1
            hits += int(rank is not None)
            reciprocal_ranks.append(1 / rank if rank else 0.0)
            course_top += int(bool(results) and results[0]["course"])
            wrongly_blocked += int(not results)
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_outcome": case["expected_outcome"],
                "hit_rank": rank,
                "retrieved": results,
            }
        )

    def ratio(part: int, whole: int) -> float | None:
        return round(part / whole, 3) if whole else None

    report = {
        "k": k,
        "reranker": settings.reranker_model or None,
        "case_count": len(cases),
        "answerable_case_count": answerable,
        "recall_at_k": ratio(hits, answerable),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3) if reciprocal_ranks else None,
        "top1_from_course": ratio(course_top, answerable),
        "answerable_wrongly_blocked": ratio(wrongly_blocked, answerable),
        "off_topic_blocked": ratio(off_topic_blocked, abstain),
        "mean_seconds": round(sum(seconds) / len(seconds), 3),
        "cases": rows,
    }
    report_path = settings.manifest_path.parent / "evaluation" / "retrieval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SubjectMate retrieval")
    parser.add_argument("--k", type=int, default=6, help="Number of chunks retrieved per question")
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")

    report = evaluate_retrieval(Settings(), args.k)
    for key in (
        "recall_at_k", "mrr", "top1_from_course", "answerable_wrongly_blocked",
        "off_topic_blocked", "mean_seconds",
    ):
        print(f"{key}: {report[key]}")
    print(f"Cases: {report['case_count']}; report: storage/evaluation/retrieval_report.json")


if __name__ == "__main__":
    main()
