#!/usr/bin/env python3
"""
Minimal **multi-turn tool-calling** demo for the Qwen2.5/3-4B Search-R1 model
(무료 소스만 사용: DuckDuckGo + Google News RSS)

개선 사항
---------
- DuckDuckGo 지역/백엔드 고정(region="kr-kr", backend="lite")로 한글 결과 향상
- 뉴스성 질의 보강: DDG.news + Google News RSS
- 한글 결과 우선 정렬(간단한 Hangul ratio)
- 단일 <tool_response> ... </tool_response> 래핑
- 예외/타임아웃/중복 제거/출력 포맷 개선
"""
from __future__ import annotations
import html
import json
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List

import requests
import torch
from duckduckgo_search import DDGS
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. Call the `thinking` tool whenever a short planning note will help. "
    "Fill the fields `goal`, `status`, and `next_step` to track your current focus, what you know so far, "
    "and the single next move. Use the `search` tool if you need external knowledge—the results will appear "
    "between <tool_response> and </tool_response>. Invoke tools as needed, then provide the final answer inside "
    "<answer> and </answer> without extra explanation. For example, <answer>Beijing</answer>. Question: "
)

# 모델 경로는 그대로 사용하거나 바꾸세요.
MODEL_NAME = "checkpoints/search_r1_like_async_rl/qwen3-4b-search-r1/global_step_60/actor/huggingface"
MAX_TURNS = 40
MAX_RESPONSE_TOKENS = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "DuckDuckGo web/news search (+ Google News RSS fallback, all free)",
        "parameters": {
            "type": "object",
            "properties": {
                "query_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fully-formed semantic queries."
                }
            },
            "required": ["query_list"],
        },
    },
}
THINKING_SCHEMA = {
    "type": "function",
    "function": {
        "name": "thinking",
        "description": "A planning tool to help you think through a problem.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "The current goal or task."},
                "status": {"type": "string", "description": "What you know so far."},
                "next_step": {"type": "string", "description": "The single next step to take."}
            },
            "required": ["goal", "status", "next_step"],
        },
    },
}

def create_prompt(q: str) -> List[dict]:
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_CONTENT},
        {"role": "user", "content": DEFAULT_USER_CONTENT_PREFIX + q},
    ]

# ---------- 검색 유틸 ----------

def _hangul_ratio(s: str) -> float:
    total = len(s)
    if total == 0:
        return 0.0
    cnt = len(re.findall(r"[가-힣]", s))
    return cnt / total

def _format_hits(hits: List[Dict]) -> str:
    if not hits:
        return "No results."
    lines = []
    for i, h in enumerate(hits, 1):
        title = (h.get("title") or "").replace("\n", " ").strip()
        snippet = (h.get("snippet") or "").replace("\n", " ").strip()
        url = (h.get("url") or "").strip()
        lines.append(f"{i}. {title} – {snippet} ({url})")
    return "\n".join(lines)

def _merge_dedup(hits: Iterable[Dict], prefer_ko: bool = True, topk: int = 5) -> List[Dict]:
    seen = set()
    merged = []
    for h in hits:
        url = (h.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        merged.append(h)
    if prefer_ko:
        merged.sort(
            key=lambda x: _hangul_ratio((x.get("title") or "") + " " + (x.get("snippet") or "")),
            reverse=True,
        )
    return merged[:topk]

def ddg_search_text(query: str, k: int = 5, region: str = "kr-kr") -> List[Dict]:
    out: List[Dict] = []
    try:
        # backend="lite"가 한글/국내 결과 비율이 높은 편
        with DDGS(timeout=10) as ddgs:
            for h in ddgs.text(query, region=region, safesearch="moderate", backend="lite", max_results=k * 2):
                out.append({"title": h.get("title"), "snippet": h.get("body"), "url": h.get("href")})
                if len(out) >= k:
                    break
    except Exception as e:
        out.append({"title": "[DDG text search error]", "snippet": str(e), "url": ""})
    return out

def ddg_search_news(query: str, k: int = 5, region: str = "kr-kr") -> List[Dict]:
    out: List[Dict] = []
    try:
        with DDGS(timeout=10) as ddgs:
            for h in ddgs.news(query, region=region, safesearch="moderate", max_results=k):
                out.append({"title": h.get("title"), "snippet": h.get("body"), "url": h.get("url")})
    except Exception as e:
        out.append({"title": "[DDG news search error]", "snippet": str(e), "url": ""})
    return out

def google_news_rss(query: str, k: int = 5, lang: str = "ko", country: str = "KR") -> List[Dict]:
    """
    무료(키 불필요) 보강. 뉴스성 키워드에서 강함.
    """
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl={lang}&gl={country}&ceid={country}:{lang}"
    out: List[Dict] = []
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for item in root.findall(".//item")[:k]:
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            desc = item.findtext("description") or ""
            out.append({"title": html.unescape(title.strip()),
                        "snippet": html.unescape(desc.strip()),
                        "url": link.strip()})
    except Exception as e:
        out.append({"title": "[Google News RSS error]", "snippet": str(e), "url": ""})
    return out

def free_search(query: str, k: int = 5) -> List[Dict]:
    """
    DDG(news + text) + Google News RSS를 합쳐 dedup 및 한글 우선 정렬.
    """
    hits: List[Dict] = []
    # 뉴스 우선
    hits.extend(ddg_search_news(query, k, region="en-us"))
    hits.extend(google_news_rss(query, k, lang="en", country="US"))
    # 일반 웹
    if len(hits) < k:
        hits.extend(ddg_search_text(query, k))
    return _merge_dedup(hits, prefer_ko=True, topk=k)

# ---------- 툴 호출 파싱 ----------

def extract_tool_call(s: str) -> Dict:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # 모델이 JSON이 아닌 텍스트를 넣는 경우를 대비한 fallback
        return {"name": "search", "arguments": {"query_list": [s]}}

def extract_queries_from_tool_call(tool_call: Dict) -> List[str]:
    if tool_call.get("name") == "search":
        args = tool_call.get("arguments", {})
        ql = args.get("query_list") or []
        if isinstance(ql, list) and ql:
            return [str(q) for q in ql]
    return []

# ---------- 메인 루프 ----------

def main() -> None:
    q = sys.argv[1] if len(sys.argv) > 1 else """
이재명 대통령 당선일은 언제야?"""

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    msgs = create_prompt(q)
    tool_pat = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)

    for turn in range(MAX_TURNS):
        inputs = tok.apply_chat_template(
            msgs,
            tools=[SEARCH_SCHEMA, THINKING_SCHEMA],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(DEVICE)

        out = model.generate(
            inputs,
            max_new_tokens=MAX_RESPONSE_TOKENS,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
        )
        new = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

        print(f"\n===== Assistant (turn {turn+1}) =====\n")
        # print(new)
        msgs.append({"role": "assistant", "content": new})

        m = tool_pat.search(new)
        if not m:
            if "<answer>" in new:
                # 결과 강조
                print("\033[93m" + new + "\033[0m")
            break

        tool_call_str = m.group(1)
        tool_call = extract_tool_call(tool_call_str)
        print("\033[92m" + json.dumps(tool_call, ensure_ascii=False, indent=2) + "\033[0m")

        results_blocks: List[str] = []
        if tool_call.get("name") == "search":
            queries = extract_queries_from_tool_call(tool_call) or [q]
            for qq in queries:
                hits = free_search(qq, k=5)
                results_blocks.append(f"# {qq}\n" + _format_hits(hits))
        elif tool_call.get("name") == "thinking":
            results_blocks.append("NOTE_ACCEPTED_PROCEED_WITH_NEXT_STEP")
        else:
            results_blocks.append(f"UNKNOWN_TOOL:{tool_call.get('name')}")

        # ★ 단일 래핑 ★
        tool_response = "<tool_response>\n" + "\n\n---\n\n".join(results_blocks) + "\n</tool_response>"
        print("\033[90m" + tool_response + "\033[0m")
        msgs.append({"role": "user", "content": tool_response})


    # push to hub
    # model.push_to_hub('Seungyoun/Qwen3-4B-search-r1-w-selective-plan')
    tok.push_to_hub('Seungyoun/Qwen3-4B-search-r1-w-selective-plan')

if __name__ == "__main__":
    main()
