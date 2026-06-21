import sys
import json
from pathlib import Path

import numpy as np
import cv2
import easyocr



def set_exit_error(message: str) -> None:
    print(message)
    sys.exit(1)


def get_image_processed(image_origin):
    image_gray = cv2.cvtColor(image_origin, cv2.COLOR_BGR2GRAY)
    return image_gray


def get_data_text(image):
    reader = easyocr.Reader(["en", "ru"], gpu=True)
    data_raw = reader.readtext(image)

    data_text_dimension = []
    data_text_chamfer = []
    for data in data_raw:
        if data[1].find('ф') == -1:
            data_text_dimension.append(data)       
            continue

        data_text_chamfer.append(data)

    return data_text_dimension, data_text_chamfer
    


def get_dict_result(path, data_text):
    result = {
        "filename": path.name,
        "unit": "mm",
        "count_dimension": {
            "all": len(data_text[0]),
            "chamfers": len(data_text[1]),
        },
        "text_raw": [text for _, text, _ in data_text[0]],
    }
    return result


def set_export(path_export, dict_result, image):
    path_export.mkdir(exist_ok=True)
    with open(path_export / "result.json", "w", encoding="utf-8") as file:
        json.dump(dict_result, file)

    cv2.imwrite(path_export / "overlay.png", image)
    print("Info: Results exported into out folder")


def get_border_text(data_raw):
    border_text = []
    for bbox, text, conf in data_raw:
        x = [p[0] for p in bbox]
        y = [p[1] for p in bbox]
        corner_tl = (int(min(x)), int(min(y)))
        corner_br = (int(max(x)), int(max(y)))
        border_text.append([corner_tl, corner_br])

    return border_text


def get_data_lines_thick(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    paper = cv2.bitwise_not(binary)

    h, w = paper.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(w):
        if paper[0, x] == 255:
            cv2.floodFill(paper, mask, (x, 0), 128)

        if paper[h - 1, x] == 255:
            cv2.floodFill(paper, mask, (x, h - 1), 128)

    for y in range(h):
        if paper[y, 0] == 255:
            cv2.floodFill(paper, mask, (0, y), 128)

        if paper[y, w - 1] == 255:
            cv2.floodFill(paper, mask, (w - 1, y), 128)

    inside = (paper == 255).astype(np.uint8) * 255
    contours, _ = cv2.findContours(inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = [c for c in contours if cv2.contourArea(c) > 200]
    result.sort(key=cv2.contourArea, reverse=True)
    return result


def set_overlay_text_outline(image, borders, color, thickness):
    for border in borders:
        cv2.rectangle(image, border[0], border[1], color, thickness)

    return image


def set_overlay_data_lines_thick(image, contours, color, thickness):
    for contour in contours:
        if cv2.arcLength(contour, True) < 40:
            continue

        cv2.drawContours(image, [contour], -1, color, thickness)


def get_image_overlay(image, data_text, data_lines_thick):
    image_overlay = image.copy()
    set_overlay_text_outline(
        image_overlay,
        get_border_text(data_text),
        (0, 0, 255),
        3,
    )
    set_overlay_data_lines_thick(
        image_overlay,
        data_lines_thick,
        (0, 255, 0),
        2
    )
    return image_overlay


def main() -> None:
    if not len(sys.argv) == 2:
        set_exit_error("Error: not exist image path")
        sys.exit(1)

    path_image = Path(sys.argv[1]).resolve()
    if not path_image.exists():
        set_exit_error("Error: image file not found")

    image_origin = cv2.imread(path_image, cv2.IMREAD_COLOR)
    image_processed = get_image_processed(image_origin)
    data_text = get_data_text(image_processed)
    data_lines_thick = get_data_lines_thick(image_processed)

    image_overlay = get_image_overlay(image_origin, data_text[0], data_lines_thick)
    dict_result = get_dict_result(path_image, data_text)
    set_export(path_image.parent / "out", dict_result, image_overlay)



if __name__ == "__main__":
    main()
