import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import _chunk_law_text, _keyword_search, enrich_prompt, format_context, search


class TestChunking(unittest.TestCase):
    def test_chunks_articles(self):
        text = (
            "제1조(목적) 이 법은 도로에서 일어나는 교통상의 위험을 방지한다.\n"
            "제2조(정의) 이 법에서 사용하는 용어의 뜻은 다음과 같다.\n"
            "  1. 도로란 도로법에 따른 도로를 말한다.\n"
            "제3조(신호) 신호기의 종류는 다음과 같다.\n"
        )
        chunks = _chunk_law_text(text, "test")
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["article_id"], "제1조")
        self.assertEqual(chunks[0]["title"], "목적")

    def test_chunks_from_real_file(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "data", "road_traffic_act.txt")
        if not os.path.exists(path):
            self.skipTest("road_traffic_act.txt not found")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        chunks = _chunk_law_text(text, "도로교통법")
        self.assertGreater(len(chunks), 100)  # ~160 articles expected


class TestKeywordSearch(unittest.TestCase):
    def test_school_zone(self):
        results = _keyword_search("어린이 보호구역 스쿨존 속도")
        self.assertGreater(len(results), 0)
        found_titles = [r.get("title", "") for r in results]
        self.assertTrue(any("보호구역" in t or "어린이" in t for t in found_titles),
                        f"Expected school zone article, got: {found_titles}")

    def test_speed_limit(self):
        results = _keyword_search("제한속도 최고속도")
        self.assertGreater(len(results), 0)

    def test_no_match(self):
        results = _keyword_search("xyzabc qqq zzz")
        self.assertEqual(len(results), 0)


class TestEnrichPrompt(unittest.TestCase):
    def test_enriches_with_context(self):
        enriched = enrich_prompt("어린이 보호구역 등하교 시간 시뮬레이션", api_key="invalid")
        self.assertIn("[참고 규정]", enriched)

    def test_enrichment_contains_original_input(self):
        original = "xyzabc qqq zzz"
        enriched = enrich_prompt(original, api_key="invalid")
        self.assertTrue(enriched.startswith(original))


class TestFormatContext(unittest.TestCase):
    def test_formats_regulations(self):
        regs = [{"source": "도로교통법", "article_id": "제12조",
                 "title": "어린이 보호구역", "text": "테스트 조문입니다."}]
        result = format_context(regs)
        self.assertIn("[도로교통법 제12조]", result)
        self.assertIn("테스트 조문입니다.", result)

    def test_empty_list(self):
        self.assertEqual(format_context([]), "")

    def test_truncates_long_text(self):
        regs = [{"source": "법", "article_id": "제1조",
                 "title": "t", "text": "가" * 500}]
        result = format_context(regs)
        self.assertIn("...", result)


if __name__ == "__main__":
    unittest.main()
