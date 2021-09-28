import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rockfish.extraction.alignment import get_aligner, align_read

import unittest


class AlignmentTest(unittest.TestCase):
    def test_alignment(self):
        with open('data/test_alignment/query.txt') as f:
            query = f.read()

        aligner = get_aligner('data/test_alignment/ref.fasta')
        aln = align_read(query, aligner)

        self.assertEqual(aln.r_start, 0)
        self

        print(aln.r_start, aln.r_end, aln.ref_to_query[0], aln.ref_to_query[-1], len(query))

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()