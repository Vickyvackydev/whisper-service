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

    def test_my_casing_normalization(self):
        cases = [
            ("This is My client and My submission and My Lord.", "This is my client and my submission and My Lord."),
            ("My client is ready. My Lord, I agree.", "My client is ready. My Lord, I agree."),
            ("Yes, My Lord. That is My argument.", "Yes, My Lord. That is my argument."),
            ("in My humble view, My Lord.", "in my humble view, My Lord."),
            ("It was My fault, My Noble Lord.", "It was my fault, My Noble Lord."),
            ("My Lords, please note My presence.", "My Lords, please note my presence."),
            ("No, My friend.", "No, my friend."),
            ("Where is My car? My car is there.", "Where is my car? My car is there."),
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                self.assertEqual(normalize_case_numbers_and_slashes(inp), expected)

        # Word level test
        seg = {
            "text": "This is My client. Yes, my lord.",
            "words": [
                {"word": "This"},
                {"word": "is"},
                {"word": "My"},
                {"word": "client."},
                {"word": "Yes,"},
                {"word": "my"},
                {"word": "lord."},
            ]
        }
        res = normalize_segment(seg)
        self.assertEqual(res["text"], "This is my client. Yes, My Lord.")
        words_out = [w["word"] for w in res["words"]]
        self.assertEqual(words_out, ["This", "is", "my", "client.", "Yes,", "My", "Lord."])

    def test_slashes_and_no_spaces(self):
        cases = [
            ("the word slash should appear as /", "the word/should appear as/"),
            ("plaintiff slash defendant", "Plaintiff/Defendant"),
            ("plaintiff / defendant", "Plaintiff/Defendant"),
            ("one slash two", "1/2"),
            ("one / two", "1/2"),
            ("fhc slash abj slash cs slash 55 slash 2024", "FHC/ABJ/CS/55/2024"),
            ("May slash June", "May/June"),
            ("and slash or", "and/or"),
            ("yes slash no", "yes/no"),
            ("Exhibit A slash 1", "Exhibit A/1"),
            ("section 10 slash 12", "section 10/12"),
            ("slash 2026", "/2026"),
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                self.assertEqual(normalize_case_numbers_and_slashes(inp), expected)

    def test_exhibit_and_exhibits_capitalization(self):
        cases = [
            ("exhibit 1", "Exhibit 1"),
            ("exhibit five", "Exhibit 5"),
            ("exhibit 5A", "Exhibit 5A"),
            ("exhibit five a", "Exhibit 5A"),
            ("exhibit A", "Exhibit A"),
            ("exhibit P1", "Exhibit P1"),
            ("exhibits 1 and 2", "Exhibits 1 and 2"),
            ("exhibits one and two", "Exhibits 1 and 2"),
            ("Please look at exhibit 5.", "Please look at Exhibit 5."),
            ("We refer to exhibits 1 and 2.", "We refer to Exhibits 1 and 2."),
            ("We tender this as an Exhibit in Court.", "We tender this as an exhibit in Court."),
            ("These Exhibits were marked.", "These exhibits were marked."),
            ("Exhibit is admitted.", "Exhibit is admitted."),
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                self.assertEqual(normalize_case_numbers_and_slashes(inp), expected)

    def test_courtroom_decorum_and_honorifics(self):
        cases = [
            ("as the court pleases", "As the Court pleases"),
            ("The matter is adjourned, as the court pleases.", "The matter is adjourned, as the Court pleases."),
            ("may it please the court", "May it please the Court"),
            ("May it please the Court, my lord.", "May it please the Court, My Lord."),
            ("if it may it please the court", "if it may it please the Court"),
            ("much obliged", "Much obliged"),
            ("We are much obliged to your lordship.", "We are much obliged to Your Lordship."),
            ("my learned colleague", "My learned colleague"),
            ("I agree with my learned colleague.", "I agree with my learned colleague."),
            ("my noble lord", "My Noble Lord"),
            ("I submit to my noble lord.", "I submit to My Noble Lord."),
            ("my noble lordship", "My Noble Lordship"),
            ("I submit to my noble lordship.", "I submit to My Noble Lordship."),
            ("no objection", "No objection"),
            ("We have no objection to the document.", "We have No objection to the document."),
            ("honourable court", "Honourable Court"),
            ("honorable court", "Honourable Court"),
            ("your honor", "Your Honor"),
            ("your honour", "Your Honour"),
            ("your ladyship", "Your Ladyship"),
            ("your ladyships", "Your Ladyships"),
            ("your lordship", "Your Lordship"),
            ("your lordships", "Your Lordships"),
            ("your worship", "Your Worship"),
            ("learned counsel", "Learned Counsel"),
            ("learned friend", "Learned Friend"),
            ("learned silk", "Learned Silk"),
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                self.assertEqual(normalize_case_numbers_and_slashes(inp), expected)

    def test_latin_maxims_and_proceedings(self):
        cases = [
            ("without prejudice", "Without prejudice"),
            ("This offer is without prejudice.", "This offer is without prejudice."),
            ("with due respect", "With due respect"),
            ("With due respect to my learned friend.", "With due respect to my Learned Friend."),
            ("in witness whereof", "in witness Whereof"),
            ("whereof", "Whereof"),
            ("certified true copy", "Certified True Copy"),
            ("cross-examination", "Cross-Examination"),
            ("cross-examine", "cross-examine"),
            ("examination-in-chief", "Examination-in-Chief"),
            ("re-examination", "Re-examination"),
            ("pre-trial", "Pre-trial"),
            ("ruling", "Ruling"),
            ("ordered as prayed", "Ordered as prayed"),
            ("The application is ordered as prayed.", "The application is ordered as prayed."),
            ("ex parte", "Ex Parte"),
            ("in limine", "In Limine"),
            ("per se", "Per Se"),
            ("allocutus", "allocutus"),
            ("estoppel", "estoppel"),
            ("statute-barred", "statute-barred"),
            ("inter alia", "inter alia"),
            ("locus in quo", "locus in quo"),
            ("locus standi", "locus standi"),
            ("mutatis mutandis", "mutatis mutandis"),
            ("obiter dictum", "obiter dictum"),
            ("prima facie", "prima facie"),
            ("ratio decidendi", "ratio decidendi"),
            ("res judicata", "res judicata"),
            ("subjudice", "subjudice"),
            ("sub judice", "subjudice"),
            ("suo motou", "suo motou"),
            ("suo motu", "suo motou"),
            ("ultra vires", "ultra vires"),
            ("aforementioned", "aforementioned"),
            ("aforesaid", "aforesaid"),
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                self.assertEqual(normalize_case_numbers_and_slashes(inp), expected)

    def test_parties_and_court_entities(self):
        cases = [
            ("federal high court", "Federal High Court"),
            ("state high court", "State High Court"),
            ("court of appeal", "Court of Appeal"),
            ("supreme court", "Supreme Court"),
            ("national industrial court", "National Industrial Court"),
            ("magistrate court", "Magistrate Court"),
            ("customary court", "Customary Court"),
            ("in chambers", "In Chambers"),
            ("attorney general", "Attorney General"),
            ("solicitor general", "Solicitor General"),
            ("director of public prosecutions", "Director of Public Prosecutions"),
            ("chief judge", "Chief Judge"),
            ("chief justice", "Chief Justice"),
            ("chief registrar", "Chief Registrar"),
            ("deputy registrar", "Deputy Registrar"),
            ("deputy sheriff", "Deputy Sheriff"),
            ("presiding judge", "Presiding Judge"),
            ("senior judge", "Senior Judge"),
            ("legal practitioner", "legal practitioner"),
            ("solicitor", "solicitor"),
            ("co-defendant", "Co-Defendant"),
            ("co-plaintiff", "Co-Plaintiff"),
            ("co-respondent", "Co-Respondent"),
            ("counter-claimant", "Counter-Claimant"),
            ("defence witness", "Defence Witness"),
            ("defense witness", "Defence Witness"),
            ("prosecution witness", "Prosecution Witness"),
            ("judgment creditor", "Judgment Creditor"),
            ("judgment debtor", "Judgment Debtor"),
            ("affidavit of service", "Affidavit of Service"),
            ("counter-affidavit", "Counter-Affidavit"),
            ("further affidavit", "Further Affidavit"),
            ("originating summons", "Originating Summons"),
            ("originating motion", "Originating Motion"),
            ("writ of summons", "Writ of Summons"),
            ("written address", "Written Address"),
            ("statement of claim", "Statement of Claim"),
            ("statement of defence", "Statement of Defence"),
            ("statement of defense", "Statement of Defence"),
            ("statement on oath", "Statement on Oath"),
            ("motion on notice", "Motion on Notice"),
            ("motion ex parte", "Motion Ex Parte"),
            ("consent judgment", "Consent Judgment"),
            ("consent order", "Consent Order"),
            ("default judgment", "Default Judgment"),
            ("interlocutory injunction", "Interlocutory Injunction"),
            ("stay of execution", "Stay of Execution"),
            ("stay of proceedings", "Stay of Proceedings"),
            ("perpetual injunction", "Perpetual Injunction"),
            ("declaratory relief", "Declaratory Relief"),
            ("record of proceedings", "record of proceedings"),
            ("subpoena", "subpoena"),
            ("gazette", "gazette"),
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                self.assertEqual(normalize_case_numbers_and_slashes(inp), expected)

    def test_word_tokens_slashed_merge_and_alignment(self):
        seg = {
            "text": "plaintiff slash defendant in exhibit 5",
            "words": [
                {"word": "plaintiff", "start": 0.0, "end": 0.4},
                {"word": "slash", "start": 0.4, "end": 0.6},
                {"word": "defendant", "start": 0.6, "end": 1.0},
                {"word": "in", "start": 1.0, "end": 1.2},
                {"word": "exhibit", "start": 1.2, "end": 1.5},
                {"word": "five", "start": 1.5, "end": 1.8},
            ]
        }
        res = normalize_segment(seg)
        self.assertEqual(res["text"], "Plaintiff/Defendant in Exhibit 5")
        words = res["words"]
        self.assertEqual(len(words), 4)
        self.assertEqual(words[0]["word"], "Plaintiff/Defendant")
        self.assertEqual(words[0]["start"], 0.0)
        self.assertEqual(words[0]["end"], 1.0)
        self.assertEqual(words[1]["word"], "in")
        self.assertEqual(words[2]["word"], "Exhibit")
        self.assertEqual(words[3]["word"], "5")

    def test_user_reported_court_rules(self):
        cases = [
            # 1. Suit number hyphen to slash conversion
            ("Suit No. E-57D-2023 between Peter and Christiana", "Suit No. E/57D/2023 between Peter and Christiana"),
            ("Motion Expatate E-447M-2023 for substituted service", "Motion Ex Parte E/447M/2023 for substituted service"),
            ("FHC-ABJ-CS-55-2024", "FHC/ABJ/CS/55/2024"),
            ("E-57D-2023", "E/57D/2023"),

            # 2. Calendar months capitalized
            ("25th of march 2025 for continuation", "25th of March 2025 for continuation"),
            ("adjourned to 10th of january 2024", "adjourned to 10th of January 2024"),
            ("in december last year", "in December last year"),

            # 3. Nigerian states capitalized
            ("High Court of enugu state", "High Court of Enugu state"),
            ("sitting in lagos", "sitting in Lagos"),
            ("matter in akwa ibom state", "matter in Akwa Ibom state"),
            ("in cross river state", "in Cross River state"),
            ("Federal Capital Territory", "Federal Capital Territory"),

            # 4. Criminal Code Law capitalized
            ("pursuant to the criminal code law", "pursuant to the Criminal Code Law"),
            ("under the criminal code", "under the Criminal Code"),
            ("under penal code law", "under Penal Code Law"),

            # 5. Objection, my Lord vs My Lord
            ("that objection, My Lord.", "that objection, my Lord."),
            ("Objection, my Lord. The evidence is hearsay.", "Objection, my Lord. The evidence is hearsay."),
            ("My Lord, Petitioner is present in court.", "My Lord, Petitioner is present in court."),

            # 6. Demand / Demands lowercase mid-sentence
            ("The defense Demands that we file today.", "The defense demands that we file today."),
            ("Demand must be served on time.", "Demand must be served on time."),
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                self.assertEqual(normalize_case_numbers_and_slashes(inp), expected)

    def test_reconstruct_speaker_turns_courtroom_alternation(self):
        from ml_worker.pipeline import InferencePipeline
        pipeline = InferencePipeline.__new__(InferencePipeline)

        segments = [
            {"start": 0.901, "end": 3.863, "text": "My Lord is giving me a new name.", "speaker": "SPEAKER_00",
             "words": [{"word": tok, "start": 0.9 + i*0.3, "end": 0.9 + (i+1)*0.3, "speaker": "SPEAKER_00"}
                       for i, tok in enumerate("My Lord is giving me a new name.".split())]},
            {"start": 5.143, "end": 8.966, "text": "Well, choose the one you want.", "speaker": "SPEAKER_00",
             "words": [{"word": tok, "start": 5.1 + i*0.4, "end": 5.1 + (i+1)*0.4, "speaker": "SPEAKER_00"}
                       for i, tok in enumerate("Well, choose the one you want.".split())]},
            {"start": 10.327, "end": 11.187, "text": "What are you talking about?", "speaker": "SPEAKER_01",
             "words": [{"word": tok, "start": 10.3 + i*0.2, "end": 10.3 + (i+1)*0.2, "speaker": "SPEAKER_01"}
                       for i, tok in enumerate("What are you talking about?".split())]},
            {"start": 11.588, "end": 11.888, "text": "My Lord,", "speaker": "SPEAKER_00",
             "words": [{"word": tok, "start": 11.5 + i*0.2, "end": 11.5 + (i+1)*0.2, "speaker": "SPEAKER_00"}
                       for i, tok in enumerate("My Lord,".split())]},
            {"start": 26.498, "end": 27.798, "text": "Can I see the prayer?", "speaker": "SPEAKER_01",
             "words": [{"word": tok, "start": 26.4 + i*0.2, "end": 26.4 + (i+1)*0.2, "speaker": "SPEAKER_01"}
                       for i, tok in enumerate("Can I see the prayer?".split())]},
            {"start": 28.339, "end": 28.539, "text": "Sir?", "speaker": "SPEAKER_00",
             "words": [{"word": "Sir?", "start": 28.3, "end": 28.5, "speaker": "SPEAKER_00"}]},
            {"start": 28.679, "end": 29.359, "text": "Can I see the order?", "speaker": "SPEAKER_01",
             "words": [{"word": tok, "start": 28.6 + i*0.2, "end": 28.6 + (i+1)*0.2, "speaker": "SPEAKER_01"}
                       for i, tok in enumerate("Can I see the order?".split())]},
            {"start": 29.68, "end": 30.1, "text": "Yes, My Lord.", "speaker": "SPEAKER_00",
             "words": [{"word": tok, "start": 29.6 + i*0.2, "end": 29.6 + (i+1)*0.2, "speaker": "SPEAKER_00"}
                       for i, tok in enumerate("Yes, My Lord.".split())]},
            {"start": 139.881, "end": 140.945, "text": "the order you asked", "speaker": "SPEAKER_01",
             "words": [{"word": tok, "start": 139.8 + i*0.2, "end": 139.8 + (i+1)*0.2, "speaker": "SPEAKER_01"}
                       for i, tok in enumerate("the order you asked".split())]},
            {"start": 140.985, "end": 141.888, "text": "for yes sir does", "speaker": "SPEAKER_00",
             "words": [{"word": tok, "start": 140.9 + i*0.2, "end": 140.9 + (i+1)*0.2, "speaker": "SPEAKER_00"}
                       for i, tok in enumerate("for yes sir does".split())]},
            {"start": 141.908, "end": 142.31, "text": "not talk about", "speaker": "SPEAKER_01",
             "words": [{"word": tok, "start": 141.9 + i*0.1, "end": 141.9 + (i+1)*0.1, "speaker": "SPEAKER_01"}
                       for i, tok in enumerate("not talk about".split())]},
            {"start": 151.643, "end": 152.739, "text": "There's no subsequent process there.", "speaker": "SPEAKER_01",
             "words": [{"word": tok, "start": 151.6 + i*0.2, "end": 151.6 + (i+1)*0.2, "speaker": "SPEAKER_01"}
                       for i, tok in enumerate("There's no subsequent process there.".split())]},
            {"start": 156.966, "end": 157.56, "text": "As the Court pleases.", "speaker": "SPEAKER_00",
             "words": [{"word": tok, "start": 156.9 + i*0.2, "end": 156.9 + (i+1)*0.2, "speaker": "SPEAKER_00"}
                       for i, tok in enumerate("As the Court pleases.".split())]},
        ]

        stitched = {"segments": segments}
        processed, num_speakers = pipeline.reconstruct_speaker_turns(stitched)

        self.assertEqual(processed[0]["text"], "My Lord is giving me a new name.")
        self.assertNotEqual(processed[1]["speaker"], processed[0]["speaker"])
        self.assertEqual(processed[1]["speaker"], processed[2]["speaker"])
        self.assertEqual(processed[3]["speaker"], processed[0]["speaker"])

if __name__ == '__main__':
    unittest.main()


