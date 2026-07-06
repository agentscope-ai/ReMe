#!/usr/bin/env python3
"""统计 results JSON 文件的实验结果"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def analyze(results_path: str):
    with open(results_path) as f:
        data = json.load(f)

    total = len(data)
    print(f"{'='*60}")
    print(f"  实验结果统计: {Path(results_path).name}")
    print(f"{'='*60}")
    print(f"\n总样本数: {total}\n")

    # ---- 按 question_type 分组统计 ----
    by_type = defaultdict(list)
    for item in data:
        by_type[item["question_type"]].append(item)

    # 表头
    print(f"{'─'*80}")
    print(f"{'Question Type':<30} {'Count':>6}  {'Agentic Acc':>12}  {'Prompted Acc':>13}")
    print(f"{'─'*80}")

    agentic_total_correct = 0
    prompted_total_correct = 0

    for qt in sorted(by_type.keys()):
        items = by_type[qt]
        n = len(items)
        a_yes = sum(1 for it in items if it["agentic_judgment"]["verdict"] == "yes")
        p_yes = sum(1 for it in items if it["prompted_judgment"]["verdict"] == "yes")
        agentic_total_correct += a_yes
        prompted_total_correct += p_yes
        a_acc = a_yes / n * 100
        p_acc = p_yes / n * 100
        print(f"{qt:<30} {n:>6}  {a_yes:>3}/{n:<3} ({a_acc:5.1f}%)  {p_yes:>3}/{n:<3} ({p_acc:5.1f}%)")

    print(f"{'─'*80}")
    a_total_acc = agentic_total_correct / total * 100
    p_total_acc = prompted_total_correct / total * 100
    print(f"{'Overall':<30} {total:>6}  {agentic_total_correct:>3}/{total:<3} ({a_total_acc:5.1f}%)  {prompted_total_correct:>3}/{total:<3} ({p_total_acc:5.1f}%)")
    print(f"{'─'*80}")

    # ---- Agentic vs Prompted 对比 ----
    print(f"\n{'='*60}")
    print("  Agentic vs Prompted 逐条对比")
    print(f"{'='*60}")

    both_correct = 0
    both_wrong = 0
    agentic_only = 0
    prompted_only = 0

    for item in data:
        a = item["agentic_judgment"]["verdict"] == "yes"
        p = item["prompted_judgment"]["verdict"] == "yes"
        if a and p:
            both_correct += 1
        elif not a and not p:
            both_wrong += 1
        elif a and not p:
            agentic_only += 1
        else:
            prompted_only += 1

    print(f"  两者都正确:           {both_correct:>4}  ({both_correct/total*100:.1f}%)")
    print(f"  两者都错误:           {both_wrong:>4}  ({both_wrong/total*100:.1f}%)")
    print(f"  仅 Agentic 正确:      {agentic_only:>4}  ({agentic_only/total*100:.1f}%)")
    print(f"  仅 Prompted 正确:     {prompted_only:>4}  ({prompted_only/total*100:.1f}%)")

    # ---- 按 question_type 分组对比 ----
    print(f"\n{'='*60}")
    print("  按 Question Type 分组 — Agentic vs Prompted 对比")
    print(f"{'='*60}")

    for qt in sorted(by_type.keys()):
        items = by_type[qt]
        n = len(items)
        bc = bw = ao = po = 0
        for item in items:
            a = item["agentic_judgment"]["verdict"] == "yes"
            p = item["prompted_judgment"]["verdict"] == "yes"
            if a and p:
                bc += 1
            elif not a and not p:
                bw += 1
            elif a and not p:
                ao += 1
            else:
                po += 1
        print(f"\n  [{qt}] (n={n})")
        print(f"    两者都正确: {bc:>3} ({bc/n*100:.1f}%)  |  两者都错误: {bw:>3} ({bw/n*100:.1f}%)")
        print(f"    仅Agentic:  {ao:>3} ({ao/n*100:.1f}%)  |  仅Prompted: {po:>3} ({po/n*100:.1f}%)")

    # ---- sessions_ingested / dreams_triggered 统计 ----
    print(f"\n{'='*60}")
    print("  Sessions & Dreams 统计")
    print(f"{'='*60}")

    sessions = [item["sessions_ingested"] for item in data]
    dreams = [item["dreams_triggered"] for item in data]

    print(f"  Sessions ingested — min: {min(sessions)}, max: {max(sessions)}, "
          f"mean: {sum(sessions)/len(sessions):.1f}")
    print(f"  Dreams triggered  — min: {min(dreams)}, max: {max(dreams)}, "
          f"mean: {sum(dreams)/len(dreams):.1f}, "
          f"non-zero: {sum(1 for d in dreams if d > 0)}")

    # ---- 错误样本分析 ----
    print(f"\n{'='*60}")
    print("  Agentic 错误样本 (前10条)")
    print(f"{'='*60}")

    wrong_items = [item for item in data if item["agentic_judgment"]["verdict"] != "yes"]
    for i, item in enumerate(wrong_items[:10]):
        print(f"\n  [{i+1}] QID: {item['question_id']}  Type: {item['question_type']}")
        print(f"      Q: {item['question'][:80]}")
        print(f"      GT:  {item['ground_truth'][:80]}")
        print(f"      Pred: {item['agentic_response'][:80]}")
        print(f"      Reason: {item['agentic_judgment']['reason'][:100]}")

    if len(wrong_items) > 10:
        print(f"\n  ... 共 {len(wrong_items)} 条错误样本，仅显示前10条")

    # ---- Prompted 各分类详细统计 ----
    print(f"\n{'='*60}")
    print("  Prompted 各分类详细统计")
    print(f"{'='*60}")

    for qt in sorted(by_type.keys()):
        items = by_type[qt]
        n = len(items)
        correct = [it for it in items if it["prompted_judgment"]["verdict"] == "yes"]
        wrong = [it for it in items if it["prompted_judgment"]["verdict"] != "yes"]
        acc = len(correct) / n * 100

        print(f"\n  ┌─ {qt} ─────────────────────────────────")
        print(f"  │ 总数: {n}   正确: {len(correct)}   错误: {len(wrong)}   准确率: {acc:.1f}%")
        print(f"  └{'─'*50}")

        if wrong:
            print(f"  错误样本 (共{len(wrong)}条):")
            for i, w in enumerate(wrong):
                gt = str(w["ground_truth"])[:70]
                pred = str(w["prompted_response"])[:70]
                reason = w["prompted_judgment"]["reason"][:100]
                print(f"    [{i+1}] QID: {w['question_id']}")
                print(f"        Q:  {w['question'][:80]}")
                print(f"        GT: {gt}")
                print(f"        Pred: {pred}")
                print(f"        Reason: {reason}")

    # ---- Prompted 汇总表格 ----
    print(f"\n{'='*60}")
    print("  Prompted 汇总表格")
    print(f"{'='*60}")
    print(f"\n  {'Category':<30} {'Total':>6} {'Correct':>8} {'Wrong':>6} {'Acc':>8}")
    print(f"  {'─'*62}")
    p_total_n = 0
    p_total_c = 0
    for qt in sorted(by_type.keys()):
        items = by_type[qt]
        n = len(items)
        c = sum(1 for it in items if it["prompted_judgment"]["verdict"] == "yes")
        w = n - c
        acc = c / n * 100
        p_total_n += n
        p_total_c += c
        print(f"  {qt:<30} {n:>6} {c:>8} {w:>6} {acc:>7.1f}%")
    print(f"  {'─'*62}")
    p_total_w = p_total_n - p_total_c
    p_total_acc = p_total_c / p_total_n * 100
    print(f"  {'Overall':<30} {p_total_n:>6} {p_total_c:>8} {p_total_w:>6} {p_total_acc:>7.1f}%")
    print(f"  {'─'*62}")

    print(f"\n{'='*60}")
    print("  统计完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "evaluation/longmemeval/results/results_s_cleaned_20260706_121027.json"
    analyze(path)
