from analysis.tests import *

phrase_detect_score = test_phrase_detect('assets/phrase_intervals.json', 30.0, 50)

print(phrase_detect_score)