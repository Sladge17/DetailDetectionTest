from pathlib import Path

import numpy as np
import pytest
import cv2

from detail_detector import (
    get_image_processed,
    get_dict_result,
)



@pytest.fixture
def sample_image():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:] = (200, 200, 200)
    cv2.putText(image, "test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return image


@pytest.fixture
def data_text():
    return (
        [([[10, 42], [60, 42], [60, 58], [10, 58]], "20", 0.95)],
        [([[80, 42], [130, 42], [130, 58], [80, 58]], "2 фаски", 0.90)],
    )



def test_get_image_processed(sample_image):
    result = get_image_processed(sample_image)
    assert len(result.shape) == 2
    assert result.shape == (100, 200)
    assert result.dtype == np.uint8


def test_get_dict_result(data_text):
    path = Path("/some/dir/image.jpg")
    result = get_dict_result(path, data_text)
    assert result["filename"] == "image.jpg"
    assert result["unit"] == "mm"
    assert result["count_dimension"]["all"] == 1
    assert result["count_dimension"]["chamfers"] == 1
    assert result["text_raw"] == ["20"]
