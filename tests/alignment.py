import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mappy
from rockfish.extraction.alignment import get_aligner, align_read

import unittest


class AlignmentTest(unittest.TestCase):
    def test_alignment(self):
        _, query, _ = list(mappy.fastx_read('tests/data/query.fa'))[0]

        aligner = get_aligner('tests/data/ref.fasta')
        aln = align_read(query, aligner)

        self.assertEqual(aln.r_start, 0)
        self.assertEqual(aln.r_end, 2204)
        self.assertEqual(aln.fwd_strand, False)
        self.assertEqual(aln.ref_to_query[0], 12)
        self.assertEqual(aln.ref_to_query[-1], 2198)
        self.assertSequenceEqual(aln.ref_to_query[:19].tolist(), [12 + i for i in range(19)])
        self.assertEqual(aln.ref_to_query[19], 31)
        self.assertSequenceEqual(aln.ref_to_query[20:119].tolist(), [31 + i for i in range(99)])
        self.assertSequenceEqual(aln.ref_to_query[-75:-73].tolist(), [2125, 2125])
        self.assertSequenceEqual(aln.ref_to_query[-74:-1].tolist(), [2198 - i for i in range(73, 0, -1)])


if __name__ == '__main__':
    unittest.main()