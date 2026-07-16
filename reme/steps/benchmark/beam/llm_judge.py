"""BEAM rubric-based LLM-as-Judge step.

For each rubric item, calls the judge LLM with the unified judge prompt
and collects scores (0.0 / 0.5 / 1.0).  The final ``llm_judge_score`` is
the average across all rubric items.

For ``event_ordering`` questions, additionally computes:
  - LLM-based event alignment (matching system events to reference events)
  - precision / recall / f1 (set-intersection after alignment)
  - Kendall's tau (ordering correlation, via scipy)
  - final_score = tau_norm * f1

A ``semantic`` alignment path is also available which uses ReMe's
configured ``as_embedding`` model (replacing BEAM's sentence_transformers).

This replicates the evaluation logic from
``datasets/BEAM/src/evaluation/compute_metrics.py``.
"""

import json
import re
from typing import List, Tuple

import numpy as np
from json_repair import repair_json
from scipy.stats import kendalltau

from ...base_step import BaseStep, Ref
from ....components import R
from ....components.as_embedding import BaseAsEmbedding
from ....enumeration import ComponentEnum


# ---------------------------------------------------------------------------
# JSON parsing helper (replicates BEAM's parse_json_response)
# ---------------------------------------------------------------------------
def _parse_json_response(response: str) -> dict:
    response = response.strip()

    if response.startswith("```"):
        match = re.search(
            r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", response, re.DOTALL,
        )
        if match:
            response = match.group(1).strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*?\}|\[.*?\])", response, re.DOTALL)
    if match:
        json_part = match.group(1)
        try:
            return json.loads(json_part)
        except Exception as e:
            raise ValueError(f"Found possible JSON but failed to parse it: {e}")

    raise ValueError("No valid JSON found in response.")


# ---------------------------------------------------------------------------
# Event-ordering helpers (replicate BEAM's compute_metrics.py)
# ---------------------------------------------------------------------------
async def _llm_equivalence(agent_wrapper, reference: str, system: str) -> bool:
    """Binary classifier: do the two snippets describe the SAME event/fact?

    Replicates BEAM's ``llm_equivalence`` using ``agent_wrapper.reply()``.
    """
    system_prompt = (
        "You are a binary classifier.\n"
        "If the TWO snippets describe the SAME event/fact, reply **YES**\n"
        "Otherwise reply **NO**. No extra words.\n"
        "DO NOT provide any explanation."
    )
    user_prompt = f"First snippet: {reference}\n\nSecond snippet: {system}"

    result = await agent_wrapper.reply(user_prompt, system_prompt=system_prompt)
    raw = (result.get("result") or "").strip().lower()
    return "yes" in raw


async def _align_with_llm(
    agent_wrapper,
    reference: List[str],
    system: List[str],
) -> Tuple[List[str], List[str]]:
    """Align system events to reference events via LLM equivalence.

    Replicates BEAM's ``align_with_llm``: for each system event, find the
    first unmatched reference event that is LLM-equivalent.  If found,
    replace the system event with the reference text (canonicalisation).
    Ensures 1-to-1 mapping.
    """
    used = set()
    system_out = []

    for s in system:
        matched_index = None
        for index, r in enumerate(reference):
            if index in used:
                continue
            if await _llm_equivalence(agent_wrapper, reference=r, system=s):
                matched_index = index
                break

        if matched_index is not None:
            system_out.append(reference[matched_index])
            used.add(matched_index)
        else:
            system_out.append(s)

    return reference, system_out


async def _semantic_align(
    embedding_fn,
    reference: List[str],
    system: List[str],
    thr: float = 0.65,
) -> Tuple[List[str], List[str]]:
    """Align system events to reference events via embedding cosine similarity.

    Replaces BEAM's ``semantic_align`` (which used sentence_transformers)
    with ReMe's configured ``as_embedding`` model.
    """
    if not reference or not system:
        return reference, system

    ref_embeddings = np.array(await embedding_fn(reference))
    sys_embeddings = np.array(await embedding_fn(system))

    # Normalise
    ref_norms = ref_embeddings / (np.linalg.norm(ref_embeddings, axis=1, keepdims=True) + 1e-12)
    sys_norms = sys_embeddings / (np.linalg.norm(sys_embeddings, axis=1, keepdims=True) + 1e-12)

    used_reference = set()
    system_canon = []

    for i, s_txt in enumerate(system):
        sims = sys_norms[i] @ ref_norms.T  # cosine similarity
        best = int(np.argmax(sims))
        if sims[best] >= thr and best not in used_reference:
            system_canon.append(reference[best])
            used_reference.add(best)
        else:
            system_canon.append(s_txt)

    return reference, system_canon


def _event_ordering_score(
    reference_canon: List[str],
    system_canon: List[str],
) -> dict:
    """Compute precision/recall/f1 + Kendall's tau after alignment.

    Replicates BEAM's ``event_ordering_score`` (the scoring part, after
    alignment is done).
    """
    tp = len(set(reference_canon) & set(system_canon))
    fp = len([x for x in system_canon if x not in reference_canon])
    fn = len([x for x in reference_canon if x not in system_canon])

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    union = list(dict.fromkeys(reference_canon + system_canon))
    tie_rank = len(union) + 1

    def to_rank(seq):
        r = {item: i + 1 for i, item in enumerate(seq)}
        return [r.get(u, tie_rank) for u in union]

    tau_b, _ = kendalltau(
        to_rank(reference_canon),
        to_rank(system_canon),
        variant="b", method="auto",
    )
    tau_b_norm = (tau_b + 1) / 2 if tau_b is not None else 0

    final_score = tau_b_norm * f1
    return dict(
        precision=precision, recall=recall, f1=f1,
        tau_norm=tau_b_norm, final_score=final_score,
    )


@R.register("beam_rubric_judge_step")
class BeamRubricJudgeStep(BaseStep):
    """Judge an LLM response against a list of rubric criteria.

    Inputs (from RuntimeContext):
        llm_response      (str, required): The model's response to evaluate.
        rubric            (list[str], required): Rubric criteria to check.
        probing_question  (str, optional): The original probing question.
        question_type     (str, optional): Question type (e.g. "event_ordering").

    Output (written to context.response):
        answer  = str(llm_judge_score)
        metadata["llm_judge_score"]      = float
        metadata["llm_judge_responses"]  = list[dict]
        metadata["event_ordering"]       = dict  (only for event_ordering type)
    """

    as_embedding: BaseAsEmbedding = Ref(
        BaseAsEmbedding, ComponentEnum.AS_EMBEDDING, optional=True,
    )

    async def execute(self):
        assert self.context is not None
        llm_response: str = self.context.get("llm_response", "")
        rubric: list[str] = self.context.get("rubric", [])
        probing_question: str = self.context.get("probing_question", "")
        question_type: str = self.context.get("question_type", "")

        if not llm_response:
            raise ValueError("beam_rubric_judge_step requires non-empty llm_response")
        if not rubric:
            raise ValueError("beam_rubric_judge_step requires non-empty rubric")
        if self.agent_wrapper is None:
            raise RuntimeError("beam_rubric_judge_step requires agent_wrapper")

        # ----- Standard rubric-based LLM-as-Judge (all question types) -----
        judge_template = self.get_prompt("judge_prompt")

        llm_judge_responses: list[dict] = []
        total_score = 0.0

        for item in rubric:
            prompt = (
                judge_template
                .replace("<rubric_item>", item)
                .replace("<llm_response>", llm_response)
            )

            result = await self.agent_wrapper.reply(prompt)
            raw = (result.get("result") or "").strip()

            try:
                parsed = _parse_json_response(raw)
            except Exception:
                try:
                    parsed = json.loads(repair_json(raw))
                except Exception:
                    parsed = {"score": 0.0, "reason": f"Failed to parse: {raw[:200]}"}

            score = float(parsed.get("score", 0))

            # Abstention: binary classification — 1.0 stays 1, <1.0 becomes 0
            if question_type == "abstention":
                score = 1.0 if score >= 1.0 else 0.0

            total_score += score
            llm_judge_responses.append(parsed)

        llm_judge_score = total_score / len(rubric) if rubric else 0.0

        self.logger.info(f"[{self.name}] judge score: {llm_judge_score:.3f}")

        self.context.response.success = True
        self.context.response.answer = str(llm_judge_score)
        self.context.response.metadata.update({
            "llm_judge_score": llm_judge_score,
            "llm_judge_responses": llm_judge_responses,
            "rubric": rubric,
            "llm_response": llm_response,
            "probing_question": probing_question,
            "question_type": question_type,
        })

        # ----- event_ordering extra metrics -----
        # Replicates BEAM's evaluate_event_ordering: system_list = llm_response.split("\n")
        # Note: BEAM calls extract_facts first but immediately overwrites with split("\n").
        if question_type == "event_ordering":
            eo_metrics = await self._compute_event_ordering(
                rubric=rubric,
                llm_response=llm_response,
            )
            self.context.response.metadata["event_ordering"] = eo_metrics
            self.logger.info(f"[{self.name}] event_ordering: {eo_metrics}")

        return self.context.response

    async def _compute_event_ordering(
        self,
        rubric: list[str],
        llm_response: str,
    ) -> dict:
        """Compute event_ordering extra metrics.

        Uses ``align_type="llm"`` to match BEAM's original code.
        Also supports ``align_type="semantic"`` via ReMe's embedding model
        (replacing sentence_transformers).
        """
        # BEAM: system_list = llm_response.split("\n")
        system_list = [line for line in llm_response.split("\n") if line.strip()]

        # Use LLM alignment (matching BEAM's align_type="llm")
        reference_canon, system_canon = await _align_with_llm(
            agent_wrapper=self.agent_wrapper,
            reference=rubric,
            system=system_list,
        )

        eo_score = _event_ordering_score(reference_canon, system_canon)

        # Also compute semantic alignment if embedding is available
        if self.as_embedding is not None:
            try:
                ref_canon_sem, sys_canon_sem = await _semantic_align(
                    embedding_fn=self.as_embedding,
                    reference=rubric,
                    system=system_list,
                )
                eo_score_sem = _event_ordering_score(ref_canon_sem, sys_canon_sem)
                eo_score["semantic_alignment"] = eo_score_sem
            except Exception as e:
                self.logger.warning(f"[{self.name}] semantic_align failed: {e}")

        return eo_score
