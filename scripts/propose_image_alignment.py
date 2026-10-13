"""Propose original-MMQA-image to parsed-figure matches for human QC.

This script reads gold assets only for evaluation alignment. Its outputs must
never be indexed or passed to an answering agent. Proposals are not gold labels.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn

from lsdrag.io import dump_json, read_jsonl, write_jsonl


def features(path):
    with Image.open(path) as image:
        rgb = ImageOps.contain(image.convert("RGB"), (1600, 1600))
        gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
        small = np.asarray(rgb.convert("L").resize((32, 32)), dtype=float)
    dct = dctn(small, norm="ortho")[:8, :8].reshape(-1)[1:]
    keypoints, descriptors = cv2.SIFT_create(nfeatures=2500).detectAndCompute(gray, None)
    return {"shape": gray.shape, "hash": dct > np.median(dct), "keypoints": keypoints, "descriptors": descriptors}


def compare(source, target):
    distance = int((source["hash"] != target["hash"]).sum())
    sh, sw = source["shape"]
    th, tw = target["shape"]
    aspect = abs(np.log((sw / sh) / (tw / th)))
    record = {"phash_distance": distance, "aspect_log_difference": float(aspect), "sift_good_matches": 0, "sift_inliers": 0, "sift_inlier_ratio": 0., "projected_corners": None, "geometric_match": False}
    a, b = source["descriptors"], target["descriptors"]
    if a is not None and b is not None and len(a) >= 4 and len(b) >= 4:
        matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a, b, k=2)
        good = [m[0] for m in matches if len(m) == 2 and m[0].distance < .7 * m[1].distance]
        record["sift_good_matches"] = len(good)
        if len(good) >= 4:
            p = np.float32([source["keypoints"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            q = np.float32([target["keypoints"][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(p, q, cv2.RANSAC, 4.)
            if matrix is not None and mask is not None:
                corners = cv2.perspectiveTransform(np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]]).reshape(-1, 1, 2), matrix).reshape(4, 2)
                inliers = int(mask.sum())
                inside = (corners[:, 0] >= -tw * .05).all() and (corners[:, 0] <= tw * 1.05).all() and (corners[:, 1] >= -th * .05).all() and (corners[:, 1] <= th * 1.05).all()
                area = abs(cv2.contourArea(corners))
                plausible = bool(np.isfinite(corners).all() and cv2.isContourConvex(corners) and inside and area > 100 and area < tw * th * 1.1)
                record.update(sift_inliers=inliers, sift_inlier_ratio=inliers / len(good), projected_corners=(corners / [tw, th]).tolist(), geometric_match=plausible and inliers >= 6 and inliers / len(good) >= .4)
    record["near_duplicate"] = bool(distance <= 4 and aspect < .08)
    record["proposal_score"] = float((1000 if record["near_duplicate"] else 0) + (100 if record["geometric_match"] else 0) + min(record["sift_inliers"], 90) - distance / 100)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--gold-sources", required=True)
    p.add_argument("--gold-images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    figures = read_jsonl(a.corpus / "figures.jsonl")
    by_url = defaultdict(list)
    for f in figures:
        by_url[f["url"]].append(f)
    a.output.mkdir(parents=True, exist_ok=True)
    cache, proposals = {}, []
    for source in read_jsonl(a.gold_sources):
        if source["url"] not in by_url:
            continue
        original = a.gold_images / Path(source["path"]).name
        sf = features(original)
        candidates = []
        for f in by_url[source["url"]]:
            key = f["image_sha256"]
            if key not in cache:
                cache[key] = features(f["image_path"])
            candidates.append({"figure_id": f["figure_id"], "page_index": f["page_index"], "document_id": f["document_id"], "image_path": f["image_path"], **compare(sf, cache[key])})
        candidates.sort(key=lambda c: (-c["proposal_score"], c["figure_id"]))
        proposals.append({"gold_image_id": source["id"], "title": source["title"], "original_path": str(original), "review_status": "unreviewed", "candidates": candidates[:3]})
    write_jsonl(a.output / "proposals.jsonl", proposals)
    for offset in range(0, len(proposals), 6):
        batch = proposals[offset:offset + 6]
        sheet = Image.new("RGB", (1440, 250 * len(batch)), "white")
        draw = ImageDraw.Draw(sheet)
        for row, proposal in enumerate(batch):
            cells = [(proposal["original_path"], "GOLD " + proposal["gold_image_id"][:12] + " " + proposal["title"])]
            cells += [(c["image_path"], c["figure_id"].split(":", 1)[-1] + f" | ph={c['phash_distance']} inliers={c['sift_inliers']}") for c in proposal["candidates"]]
            for column, (path, label) in enumerate(cells):
                with Image.open(path) as image:
                    thumbnail = ImageOps.contain(image.convert("RGB"), (350, 215))
                sheet.paste(thumbnail, (column * 360, row * 250))
                draw.text((column * 360, row * 250 + 218), label[:55], fill="black")
        sheet.save(a.output / f"review_{offset // 6:03d}.jpg", quality=90)
    report = {"gold_images_with_candidate_documents": len(proposals), "proposed_near_duplicates": sum(bool(x["candidates"] and x["candidates"][0]["near_duplicate"]) for x in proposals), "proposed_geometric_matches": sum(bool(x["candidates"] and x["candidates"][0]["geometric_match"]) for x in proposals), "labels_verified": 0, "warning": "Proposals are unverified and must not be treated as ground truth."}
    dump_json(a.output / "summary.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
