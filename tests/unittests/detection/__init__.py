import os

from unittests import _PATH_ALL_TESTS

_SAMPLE_DETECTION_SEGMENTATION = os.path.join(
    _PATH_ALL_TESTS, "_data", "detection", "instance_segmentation_inputs.json"
)
_DETECTION_VAL = os.path.join(_PATH_ALL_TESTS, "_data", "detection", "instances_val2014_100.json")
_DETECTION_BBOX = os.path.join(_PATH_ALL_TESTS, "_data", "detection", "instances_val2014_fakebbox100_results.json")
_DETECTION_SEGM = os.path.join(_PATH_ALL_TESTS, "_data", "detection", "instances_val2014_fakesegm100_results.json")
_DETECTION_OKS = os.path.join(
    _PATH_ALL_TESTS, "_data", "detection", "person_keypoints_val2014_fakekeypoints100_results.json"
)
_DETECTION_OKS_VAL = os.path.join(_PATH_ALL_TESTS, "_data", "detection", "keypoint2014_val.json")
