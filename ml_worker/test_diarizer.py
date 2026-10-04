import unittest
from ml_worker.diarizer import SpeakerDiarizer

class TestSpeakerDiarizer(unittest.TestCase):
    def setUp(self):
        self.diarizer = SpeakerDiarizer()

    def test_empty_turns_defaults_speaker_00(self):
        segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello world", "words": [{"word": "Hello", "start": 0.0, "end": 1.0}, {"word": "world", "start": 1.0, "end": 2.0}]}
        ]
        res, count = self.diarizer.assign_speakers(segments, [])
        self.assertEqual(count, 1)
        self.assertEqual(res[0]["speaker"], "SPEAKER_00")

    def test_consecutive_speakers_clean_division(self):
        """
        Speaker A speaks 0.0-3.0s.
        Speaker B speaks 3.2-6.0s.
        Words from both speakers within a single segment should be partitioned cleanly.
        """
        segments = [
            {
                "start": 0.0,
                "end": 6.0,
                "text": "Thank you for the update. We agree completely.",
                "words": [
                    {"word": "Thank", "start": 0.5, "end": 0.9},
                    {"word": "you", "start": 1.0, "end": 1.2},
                    {"word": "for", "start": 1.3, "end": 1.5},
                    {"word": "the", "start": 1.6, "end": 1.8},
                    {"word": "update.", "start": 1.9, "end": 2.5},
                    # 700ms pause between 2.5s and 3.2s
                    {"word": "We", "start": 3.2, "end": 3.5},
                    {"word": "agree", "start": 3.6, "end": 4.0},
                    {"word": "completely.", "start": 4.1, "end": 4.8},
                ]
            }
        ]
        turns = [
            {"start": 0.4, "end": 2.6, "speaker": "A"},
            {"start": 3.1, "end": 5.0, "speaker": "B"}
        ]

        res, count = self.diarizer.assign_speakers(segments, turns)
        self.assertEqual(count, 2)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["speaker"], "SPEAKER_00")
        self.assertEqual(res[0]["text"], "Thank you for the update.")
        self.assertEqual(res[1]["speaker"], "SPEAKER_01")
        self.assertEqual(res[1]["text"], "We agree completely.")

    def test_single_speaker_pausing_stays_single_bank(self):
        """
        Speaker A speaks, pauses to think, and continues speaking.
        Should remain ONE speaker bank (count=1).
        """
        segments = [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "First part of the argument.",
                "words": [
                    {"word": "First", "start": 0.5, "end": 0.8},
                    {"word": "part", "start": 0.9, "end": 1.2},
                    {"word": "of", "start": 1.3, "end": 1.4},
                    {"word": "the", "start": 1.5, "end": 1.7},
                    {"word": "argument.", "start": 1.8, "end": 2.5},
                ]
            },
            {
                "start": 4.5,
                "end": 7.0,
                "text": "Second part of the argument.",
                "words": [
                    {"word": "Second", "start": 4.8, "end": 5.2},
                    {"word": "part", "start": 5.3, "end": 5.5},
                    {"word": "of", "start": 5.6, "end": 5.7},
                    {"word": "the", "start": 5.8, "end": 6.0},
                    {"word": "argument.", "start": 6.1, "end": 6.7},
                ]
            }
        ]
        turns = [
            {"start": 0.4, "end": 2.6, "speaker": "SPEAKER_00"},
            {"start": 4.6, "end": 6.8, "speaker": "SPEAKER_00"}
        ]

        res, count = self.diarizer.assign_speakers(segments, turns)
        self.assertEqual(count, 1)
        self.assertEqual(res[0]["speaker"], "SPEAKER_00")

    def test_mid_sentence_diarized_interjection_preserved(self):
        """
        Speaker A speaks. A single word in the middle matches a distinct Pyannote turn for Speaker B.
        Diarized interjection should be preserved rather than destroyed by aggressive smoothing.
        """
        segments = [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "I was going to the court yesterday.",
                "words": [
                    {"word": "I", "start": 0.2, "end": 0.4},
                    {"word": "was", "start": 0.5, "end": 0.8},
                    {"word": "going", "start": 0.9, "end": 1.3}, # overlaps turn B
                    {"word": "to", "start": 1.4, "end": 1.6},
                    {"word": "the", "start": 1.7, "end": 1.9},
                    {"word": "court", "start": 2.0, "end": 2.4},
                    {"word": "yesterday.", "start": 2.5, "end": 3.0},
                ]
            }
        ]
        turns = [
            {"start": 0.1, "end": 0.85, "speaker": "SPEAKER_00"},
            {"start": 0.9, "end": 1.35, "speaker": "SPEAKER_01"}, # 0.45s interjection
            {"start": 1.38, "end": 3.1, "speaker": "SPEAKER_00"}
        ]

        res, count = self.diarizer.assign_speakers(segments, turns)
        self.assertEqual(count, 2)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0]["speaker"], "SPEAKER_00")
        self.assertEqual(res[1]["speaker"], "SPEAKER_01")
        self.assertEqual(res[2]["speaker"], "SPEAKER_00")

    def test_word_in_breath_pause_assigned_via_proximity(self):
        """
        A word starts 150ms before Pyannote turn starts (Whisper timestamp offset).
        It should be assigned to Speaker B via proximity, not leaked to Speaker A.
        """
        segments = [
            {
                "start": 0.0,
                "end": 8.0,
                "text": "First speaker. Now second speaker talks.",
                "words": [
                    {"word": "First", "start": 0.5, "end": 0.9},
                    {"word": "speaker.", "start": 1.0, "end": 1.5},
                    # 3.0s pause
                    # "Now" is at 4.6-4.9s. Turn for B starts at 5.0s.
                    {"word": "Now", "start": 4.6, "end": 4.9},
                    {"word": "second", "start": 5.0, "end": 5.3},
                    {"word": "speaker", "start": 5.4, "end": 5.7},
                    {"word": "talks.", "start": 5.8, "end": 6.2},
                ]
            }
        ]
        turns = [
            {"start": 0.4, "end": 1.6, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 6.5, "speaker": "SPEAKER_01"}
        ]

        res, count = self.diarizer.assign_speakers(segments, turns)
        self.assertEqual(count, 2)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["speaker"], "SPEAKER_00")
        self.assertEqual(res[0]["text"], "First speaker.")
        self.assertEqual(res[1]["speaker"], "SPEAKER_01")
        self.assertEqual(res[1]["text"], "Now second speaker talks.")

    def test_sequential_distinct_clusters_never_cross_merged(self):
        """
        Cross-speaker consolidation is disabled: Pyannote clusters are the source of truth.
        Three sequential blocks with distinct labels (e.g. Judge, Counsel, Witness each
        speaking once) must stay as 3 distinct speakers.
        """
        turns = [
            {"start": 0.0, "end": 11.2, "speaker": "SPEAKER_00"},
            {"start": 11.5, "end": 22.1, "speaker": "SPEAKER_01"},
            {"start": 22.5, "end": 33.0, "speaker": "SPEAKER_02"},
        ]
        consolidated = self.diarizer._consolidate_fragmented_speakers(turns)
        unique_spks = set(t["speaker"] for t in consolidated)
        self.assertEqual(unique_spks, {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"})

    def test_same_speaker_stitched_only_within_0_10s(self):
        turns = [
            {"start": 0.0, "end": 2.0, "speaker": "A"},
            {"start": 2.05, "end": 4.0, "speaker": "A"},  # 0.05s gap -> stitched
            {"start": 4.3, "end": 6.0, "speaker": "A"},   # 0.30s gap -> kept separate
        ]
        consolidated = self.diarizer._consolidate_fragmented_speakers(turns)
        self.assertEqual(len(consolidated), 2)
        self.assertEqual(consolidated[0]["end"], 4.0)

    def test_multi_speaker_dialogue_preserved(self):
        """
        In a real courtroom dialogue, speakers alternate (Judge -> Counsel -> Judge -> Counsel).
        This must NEVER be merged into a single speaker.
        """
        turns = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},  # Judge
            {"start": 5.5, "end": 15.0, "speaker": "SPEAKER_01"}, # Counsel
            {"start": 15.5, "end": 20.0, "speaker": "SPEAKER_00"},# Judge returns
            {"start": 20.5, "end": 35.0, "speaker": "SPEAKER_01"},# Counsel returns
        ]
        consolidated = self.diarizer._consolidate_fragmented_speakers(turns)
        unique_spks = set(t["speaker"] for t in consolidated)
        self.assertEqual(len(unique_spks), 2, f"Expected 2 speakers preserved, got: {unique_spks}")

    def test_short_interjection_preserved_in_dialogue(self):
        """
        One speaker speaks for 20s, another interjects for 0.8s ("Bring money"), then the first speaker continues.
        The 0.8s interjection must be preserved as SPEAKER_01, not absorbed by dominant speaker.
        """
        turns = [
            {"start": 0.0, "end": 20.0, "speaker": "SPEAKER_00"},
            {"start": 20.5, "end": 21.3, "speaker": "SPEAKER_01"}, # 0.8s interjection
            {"start": 22.0, "end": 45.0, "speaker": "SPEAKER_00"},
        ]
        consolidated = self.diarizer._consolidate_fragmented_speakers(turns)
        unique_spks = set(t["speaker"] for t in consolidated)
        self.assertEqual(len(unique_spks), 2)
        self.assertIn("SPEAKER_00", unique_spks)
        self.assertIn("SPEAKER_01", unique_spks)

class TestReconstructSpeakerTurns(unittest.TestCase):
    def setUp(self):
        from ml_worker.pipeline import InferencePipeline
        self.pipeline = InferencePipeline(None, None)

    def test_interjections_preserved_without_smoothing_erasure(self):
        """
        Verify that 2-word interjections ('No objection.') are preserved as distinct speaker
        turns rather than being absorbed by Pass 1 or Pass 2b.
        """
        stitched = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 4.5,
                    "text": "We seek an adjournment. No objection. Thank you my Lord.",
                    "words": [
                        {"word": "We", "start": 0.1, "end": 0.4, "speaker": "SPEAKER_00"},
                        {"word": "seek", "start": 0.5, "end": 0.8, "speaker": "SPEAKER_00"},
                        {"word": "an", "start": 0.9, "end": 1.1, "speaker": "SPEAKER_00"},
                        {"word": "adjournment.", "start": 1.2, "end": 1.8, "speaker": "SPEAKER_00"},
                        {"word": "No", "start": 2.0, "end": 2.3, "speaker": "SPEAKER_01"},
                        {"word": "objection.", "start": 2.4, "end": 2.8, "speaker": "SPEAKER_01"},
                        {"word": "Thank", "start": 3.0, "end": 3.3, "speaker": "SPEAKER_00"},
                        {"word": "you", "start": 3.4, "end": 3.6, "speaker": "SPEAKER_00"},
                        {"word": "my", "start": 3.7, "end": 3.9, "speaker": "SPEAKER_00"},
                        {"word": "Lord.", "start": 4.0, "end": 4.4, "speaker": "SPEAKER_00"},
                    ]
                }
            ]
        }
        turns, num_spks = self.pipeline.reconstruct_speaker_turns(stitched)
        self.assertEqual(num_spks, 2)
        self.assertEqual(len(turns), 3)
        self.assertEqual(turns[0]["speaker"], "SPEAKER_00")
        self.assertEqual(turns[0]["text"], "We seek an adjournment.")
        self.assertEqual(turns[1]["speaker"], "SPEAKER_01")
        self.assertEqual(turns[1]["text"], "No objection.")
        self.assertEqual(turns[2]["speaker"], "SPEAKER_00")
        self.assertEqual(turns[2]["text"], "Thank you My Lord.")

    def test_no_majority_vote_erasure_in_whisper_segment(self):
        """
        Verify that Pass 0 majority vote (which used to erase speakers with < 40% of words)
        is disabled, so a speaker with 35% of words is preserved.
        """
        stitched = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 15.0,
                    "text": "Speaker A speaks the majority of the time here. Now Speaker B answers back clearly.",
                    "words": [
                        {"word": "Speaker", "start": 0.5, "end": 0.9, "speaker": "SPEAKER_00"},
                        {"word": "A", "start": 1.0, "end": 1.2, "speaker": "SPEAKER_00"},
                        {"word": "speaks", "start": 1.3, "end": 1.6, "speaker": "SPEAKER_00"},
                        {"word": "the", "start": 1.7, "end": 1.8, "speaker": "SPEAKER_00"},
                        {"word": "majority", "start": 1.9, "end": 2.4, "speaker": "SPEAKER_00"},
                        {"word": "of", "start": 2.5, "end": 2.6, "speaker": "SPEAKER_00"},
                        {"word": "the", "start": 2.7, "end": 2.8, "speaker": "SPEAKER_00"},
                        {"word": "time", "start": 2.9, "end": 3.2, "speaker": "SPEAKER_00"},
                        {"word": "here.", "start": 3.3, "end": 3.7, "speaker": "SPEAKER_00"},
                        # Speaker B speaks 5 words (35.7% of 14 words)
                        {"word": "Now", "start": 4.5, "end": 4.8, "speaker": "SPEAKER_01"},
                        {"word": "Speaker", "start": 4.9, "end": 5.2, "speaker": "SPEAKER_01"},
                        {"word": "B", "start": 5.3, "end": 5.5, "speaker": "SPEAKER_01"},
                        {"word": "answers", "start": 5.6, "end": 6.0, "speaker": "SPEAKER_01"},
                        {"word": "back.", "start": 6.1, "end": 6.5, "speaker": "SPEAKER_01"},
                    ]
                }
            ]
        }
        turns, num_spks = self.pipeline.reconstruct_speaker_turns(stitched)
        self.assertEqual(num_spks, 2)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["speaker"], "SPEAKER_00")
        self.assertEqual(turns[1]["speaker"], "SPEAKER_01")
        self.assertEqual(turns[1]["text"], "Now Speaker B answers back.")

if __name__ == "__main__":
    unittest.main()
