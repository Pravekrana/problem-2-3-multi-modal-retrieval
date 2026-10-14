"""Video frame index and natural-language multimodal retrieval with CLIP.

Uses cosine search over L2-normalized image/text embeddings in a shared CLIP space.
Stores frame metadata in JSON and vectors in NumPy NPZ; no remote vector store required.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

def iter_videos(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in EXTENSIONS)

def frames_from_video(path, interval=2.0, max_frames=300):
    """Yield (timestamp_seconds, RGB PIL image). Seek at fixed time intervals."""
    if interval <= 0 or max_frames < 1:
        raise ValueError("interval must be positive and max_frames >= 1")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not fps or not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    frame_step = max(1, round(interval * fps))
    try:
        for number in range(max_frames):
            frame_idx = number * frame_step
            if length > 0 and frame_idx >= length:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            yield round(frame_idx / fps, 3), Image.fromarray(rgb)
    finally:
        cap.release()

def normalize(matrix):
    arr = np.asarray(matrix, dtype=np.float32)
    return arr / np.maximum(np.linalg.norm(arr, axis=-1, keepdims=True), 1e-12)

class ClipEmbedder:
    """Single model for image and text representations."""
    def __init__(self, model_id="openai/clip-vit-base-patch32", device="cpu"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = torch.device(device)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model_id = model_id

    def images(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with self.torch.inference_mode():
            values = self.model.get_image_features(**inputs)
        if not isinstance(values, self.torch.Tensor):
            values = values.pooler_output
        return normalize(values.cpu().numpy())

    def texts(self, texts):
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with self.torch.inference_mode():
            values = self.model.get_text_features(**inputs)
        if not isinstance(values, self.torch.Tensor):
            values = values.pooler_output
        return normalize(values.cpu().numpy())

def index_videos(input_path, index_path, embedder, interval=2.0, batch_size=16, max_frames=300):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    video_files = iter_videos(input_path)
    if not video_files:
        raise ValueError("No input videos found")
    metadata = []
    vectors = []
    pending_images, pending_meta = [], []
    def flush():
        if pending_images:
            encoded = embedder.images(pending_images)
            if len(encoded) != len(pending_meta):
                raise RuntimeError("Embedding batch size mismatch")
            vectors.extend(encoded)
            metadata.extend(pending_meta)
            pending_images.clear()
            pending_meta.clear()
    for path in video_files:
        try:
            for timestamp, image in frames_from_video(path, interval, max_frames):
                pending_images.append(image)
                pending_meta.append({"video": str(path), "timestamp_seconds": timestamp})
                if len(pending_images) >= batch_size:
                    flush()
        except ValueError as err:
            print(f"Skipping {path}: {err}", flush=True)
    flush()
    if not vectors:
        raise ValueError("No frames could be embedded")
    array = normalize(np.asarray(vectors, dtype=np.float32))
    folder = Path(index_path)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "vectors.npz", embeddings=array)
    manifest = {
        "model_id": embedder.model_id,
        "interval_seconds": interval,
        "frame_count": len(metadata),
        "embedding_dimensions": int(array.shape[1]),
        "frames": metadata,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    return {"indexed_frames":len(metadata), "indexed_videos":len(set(x["video"] for x in metadata)), "index":str(folder)}

def search_index(index_path, query, embedder, top_k=5, per_video=3):
    if top_k < 1 or per_video < 1:
        raise ValueError("top_k and per_video must be positive")
    folder = Path(index_path)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf8"))
    if manifest["model_id"] != embedder.model_id:
        raise ValueError("Embedding model mismatch: rebuild index or use matching model")
    with np.load(folder / "vectors.npz", allow_pickle=False) as saved:
        vectors = saved["embeddings"]
    if len(vectors) != len(manifest["frames"]):
        raise ValueError("Index and metadata length mismatch")
    if not query.strip():
        raise ValueError("Query must not be empty")
    start = time.perf_counter()
    vector = embedder.texts([query])[0]
    similarities = vectors @ vector
    ranks = np.argsort(-similarities, kind="stable")
    results, counts = [], {}
    for idx in ranks:
        item = manifest["frames"][int(idx)]
        video = item["video"]
        if counts.get(video, 0) >= per_video:
            continue
        counts[video] = counts.get(video, 0) + 1
        results.append({**item, "cosine_similarity":round(float(similarities[idx]), 5)})
        if len(results) >= top_k:
            break
    return {"query":query,"results":results,"latency_ms":round((time.perf_counter()-start)*1000,3),
            "total_indexed_frames":len(vectors)}

def main():
    parser = argparse.ArgumentParser(description="CLIP-powered text-to-video frame retrieval")
    subs = parser.add_subparsers(dest="mode", required=True)
    pindex = subs.add_parser("index")
    pindex.add_argument("--input", required=True, help="Video or directory")
    pindex.add_argument("--index", default="index")
    pindex.add_argument("--interval", type=float, default=2.0)
    pindex.add_argument("--batch-size", type=int, default=16)
    pindex.add_argument("--max-frames", type=int, default=300)
    psearch = subs.add_parser("search")
    psearch.add_argument("--index", default="index")
    psearch.add_argument("--query", required=True)
    psearch.add_argument("--top-k", type=int, default=5)
    psearch.add_argument("--per-video", type=int, default=3)
    for p in (pindex, psearch):
        p.add_argument("--model", default="openai/clip-vit-base-patch32")
        p.add_argument("--device", choices=["cpu","cuda"], default="cpu")
        p.add_argument("--output", default="results.json")
    args = parser.parse_args()
    embedder = ClipEmbedder(args.model, args.device)
    if args.mode == "index":
        report = index_videos(args.input,args.index,embedder,args.interval,args.batch_size,args.max_frames)
    else:
        report = search_index(args.index,args.query,embedder,args.top_k,args.per_video)
    Path(args.output).write_text(json.dumps(report,indent=2),encoding="utf8")
    print(json.dumps(report,indent=2))

if __name__ == "__main__":
    main()
