import json
import numpy as np
import pytest
from pathlib import Path
from src.main import normalize, iter_videos, search_index

class FakeEmbedder:
    model_id="test-model"
    def texts(self, texts):
        return normalize([[1,0]])

def test_normalize():
    x=normalize([[3,4],[0,0]])
    assert np.allclose(x[0],[.6,.8])
    assert np.isfinite(x).all()

def test_files(tmp_path):
    (tmp_path/"a.mp4").touch()
    (tmp_path/"b.txt").touch()
    assert len(iter_videos(tmp_path)) == 1

def test_search(tmp_path):
    np.savez_compressed(tmp_path/"vectors.npz",embeddings=np.array([[1,0],[0,1]],dtype=np.float32))
    (tmp_path/"manifest.json").write_text(json.dumps({
        "model_id":"test-model","frames":[
            {"video":"a.mp4","timestamp_seconds":0},
            {"video":"b.mp4","timestamp_seconds":2}]}))
    r=search_index(tmp_path,"red circle",FakeEmbedder(),top_k=1)
    assert r["results"][0]["video"]=="a.mp4"
    assert r["results"][0]["cosine_similarity"]==1.0

def test_model_mismatch(tmp_path):
    (tmp_path/"manifest.json").write_text(json.dumps({"model_id":"different"}))
    with pytest.raises(ValueError,match="mismatch"):
        search_index(tmp_path,"query",FakeEmbedder())
