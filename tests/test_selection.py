import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("prepare", Path(__file__).parents[1] / "scripts/prepare_metadata.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class SelectionTests(unittest.TestCase):
    def test_sampling_does_not_depend_on_input_order_or_answers(self):
        rows = [{"qid": str(i), "answers": [str(i)], "supporting_context": [{"doc_part": "image"}], "metadata": {"type": "a" if i < 80 else "b"}} for i in range(100)]
        a = prepare.sample_stratified(rows, 25, 17)
        changed = [dict(row, answers=["changed"]) for row in reversed(rows)]
        b = prepare.sample_stratified(changed, 25, 17)
        self.assertEqual([q["qid"] for q in a], [q["qid"] for q in b])
        self.assertEqual(sum(q["metadata"]["type"] == "a" for q in a), 20)

    def test_insufficient_candidates_rejected(self):
        with self.assertRaises(ValueError):
            prepare.sample_stratified([], 200, 17)


if __name__ == "__main__":
    unittest.main()
