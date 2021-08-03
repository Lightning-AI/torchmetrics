import numpy as np
import pytest

from torchmetrics.functional import bertscore
from torchmetrics.utilities.imports import _BERTSCORE_AVAILABLE
from torchmetrics.text import BERTScore


class CustomAssertions:
    def assertTensorsAlmostEqual(self, expected, actual, decimal=5):
        """
        Test tensors are almost equal (EPS = 1e-5 by default)
        """
        np.testing.assert_almost_equal(expected, actual, decimal=decimal)


preds = [
    "28-year-old chef found dead in San Francisco mall",
    "A 28-year-old chef who recently moved to San Francisco was "
    "found dead in the staircase of a local shopping center.",
    "The victim's brother said he cannot imagine anyone who would want to harm him,\"Finally, it went uphill again at "
    'him."',
]
refs = [
    "28-Year-Old Chef Found Dead at San Francisco Mall",
    "A 28-year-old chef who had recently moved to San Francisco was found dead in the stairwell of a local mall this "
    "week.",
    "But the victim's brother says he can't think of anyone who would want to hurt him, saying, \"Things were finally "
    'going well for him."',
]


@pytest.mark.parametrize(
    "preds,refs",
    [(preds, refs)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score_fn(preds, refs):
    Score = bertscore(preds, refs, model_type="roberta-large", num_layers=17, idf=False, batch_size=3)
    P, R, F, _ = Score["P"], Score["R"], Score["F"], Score["hash_code"]

    CustomAssertions.assertTensorsAlmostEqual(P, [0.9843302369117737, 0.9832239747047424, 0.9120386242866516])
    CustomAssertions.assertTensorsAlmostEqual(R, [0.9823839068412781, 0.9732863903045654, 0.920428991317749])
    CustomAssertions.assertTensorsAlmostEqual(F, [0.9833561182022095, 0.9782299995422363, 0.916214644908905])

@pytest.mark.parametrize(
    "preds,refs",
    [(preds, refs)],
)
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
def test_score(preds, refs):

    Scorer = BERTScore(model_type="roberta-large", num_layers=17, idf=False, batch_size=3)
    Scorer.update(predictions=preds,references=refs)
    Score = Scorer.compute()
    P, R, F, _ = Score["P"], Score["R"], Score["F"], Score["hash_code"]

    CustomAssertions.assertTensorsAlmostEqual(P, [0.9843302369117737, 0.9832239747047424, 0.9120386242866516])
    CustomAssertions.assertTensorsAlmostEqual(R, [0.9823839068412781, 0.9732863903045654, 0.920428991317749])
    CustomAssertions.assertTensorsAlmostEqual(F, [0.9833561182022095, 0.9782299995422363, 0.916214644908905])