import unittest
from ml_worker.normalizer import normalize_segment, normalize_case_numbers_and_slashes

class TestNormalizer(unittest.TestCase):
    def test_suit_number_normalization(self):
        cases = [
            ("suit number five", "Suit No. 5"),
            ("suit number 5", "Suit No. 5"),
            ("suit number 5A", "Suit No. 5A"),
            ("suit number five a", "Suit No. 5A"),
            ("suit number 5 a", "Suit No. 5A"),
            ("suits number 5", "Suit No. 5"),
            ("suits number five a", "Suit No. 5A"),
            ("suit no five", "Suit No. 5"),
            ("suit number fhc / l / cs / 485 / 2026", "Suit No. FHC/L/CS/485/2026"),
            ("case number five", "Case No. 5"),
            ("charge number five", "Charge No. 5"),
            ("appeal number five", "Appeal No. 5"),
            ("matter number five", "Matter No. 5"),
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                self.assertEqual(normalize_case_numbers_and_slashes(inp), expected)

    def test_number_letter_combinations(self):
        cases = [
            ("court says 5A it sends five a", "court says 5A it sends 5A"),
            ("5 a", "5A"),
            ("5 A", "5A"),
            ("five a", "5A"),
            ("five A", "5A"),
            ("5-a", "5A"),
            ("five-a", "5A"),
            ("one a", "1A"),
            ("1 a", "1A"),
            ("two b", "2B"),
            ("2 b", "2B"),
            ("three c", "3C"),
            ("four d", "4D"),
            ("five b", "5B"),
            ("10 a", "10A"),
            ("12 c", "12C"),
            ("five a day", "five a day"),  # preserved because 'day' is a rate/duration noun
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                self.assertEqual(normalize_case_numbers_and_slashes(inp), expected)

    def test_court_prefixes_and_numbers(self):
        cases = [
            ("Court five", "Court 5"),
            ("Court five A", "Court 5A"),
            ("Court 5 a", "Court 5A"),
            ("Room five", "Room 5"),
            ("Order five", "Order 5"),
            ("Rule six", "Rule 6"),
            ("Exhibit five", "Exhibit 5"),
            ("Exhibit five a", "Exhibit 5A"),
            ("Exhibit 5 a", "Exhibit 5A"),
            ("number five", "No. 5"),
            ("No. five", "No. 5"),
            ("paragraph five", "paragraph 5"),
            ("section five", "section 5"),
            ("count one", "count 1"),
            ("page five", "page 5"),
            ("Order eight rule six d", "Order 8 Rule 6(d)"),
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                self.assertEqual(normalize_case_numbers_and_slashes(inp), expected)

    def test_spoken_years_and_digit_series(self):
        cases = [
            ("twenty nineteen", "2019"),
            ("twenty twenty-four", "2024"),
            ("twenty twenty-five", "2025"),
            ("twenty twenty-six", "2026"),
            ("one nine zero four", "1904"),
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                self.assertEqual(normalize_case_numbers_and_slashes(inp), expected)

    def test_normalize_segment_with_words(self):
        seg = {
            "text": "suit number five a in Court five",
            "words": [
                {"word": "suit", "start": 0.0, "end": 0.2},
                {"word": "number", "start": 0.2, "end": 0.5},
                {"word": "five", "start": 0.5, "end": 0.8},
                {"word": "a", "start": 0.8, "end": 1.0},
                {"word": "in", "start": 1.0, "end": 1.2},
                {"word": "Court", "start": 1.2, "end": 1.4},
                {"word": "five", "start": 1.4, "end": 1.6},
            ]
        }
        res = normalize_segment(seg)
        self.assertEqual(res["text"], "Suit No. 5A in Court 5")
        
        words = res["words"]
        self.assertEqual(len(words), 6)
        self.assertEqual(words[0]["word"], "Suit")
        self.assertEqual(words[1]["word"], "No.")
        self.assertEqual(words[2]["word"], "5A")
        self.assertEqual(words[2]["start"], 0.5)
        self.assertEqual(words[2]["end"], 1.0)
        self.assertEqual(words[3]["word"], "in")
        self.assertEqual(words[4]["word"], "Court")
        self.assertEqual(words[5]["word"], "5")

if __name__ == '__main__':
    unittest.main()
