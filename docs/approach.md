# Multi-modal retrieval: practical design

## User problem
Retrieve specific moments from videos through natural-language requests such as “show the red object” or “find a meeting with several people.” Result granularity is **timestamped video frames**. The prototype does not provide action/event reasoning across frames, nor speech or audio retrieval.

## Architecture
Video collection -> OpenCV frame seeking at fixed temporal interval -> PIL RGB -> shared CLIP image embeddings -> L2 normalization -> compressed NumPy vectors plus JSON frame metadata -> user text -> CLIP text embedding -> cosine similarity matrix multiplication -> top-k timestamped results.

Index and query use the exact same pretrained CLIP model identifier; mismatches are rejected. Indexed frames retain source video and timestamp. Batched image embedding reduces overhead. Per-video caps prevent the top results from being exclusively one video. Index files avoid Python pickle loading and can be moved as a unit.

## Commands
`python src/demo.py` makes two synthetic test clips.
`python -m src.main index --input demo_videos --index index --interval 1` indexes frames.
`python -m src.main search --index index --query "a red circular object"` retrieves matching frames.
Use `--input /path/to/real/videos` for real footage.

## Evaluation
Create manually annotated query-to-ground-truth video/time-window judgments on a separate real collection. For each query calculate Recall@1/5/10, MRR and nDCG, scoring a result as relevant when its video and timestamp fall within the labeled time interval. Report search p50/p95 and indexing throughput, peak RAM, video hours indexed and approximate storage bytes/frame. **No real-world retrieval score is claimed here.** The included synthetic videos and mock-vector unit tests are smoke tests, not accuracy evidence.

## Cost and scale
An exact float32 search costs O(Nd) compute and 4Nd bytes; 1M frames x 512-dim is ~2.05 GB before metadata. Replace NumPy scan with FAISS IVF/HNSW/quantized index when scale justifies approximate search, and partition by tenant/camera/time. Use frame deduplication, adaptive scene cuts, thumbnails and metadata filters to reduce index size. Validate approximate recall versus exact baseline. Batch offline indexing, persist checkpoints, and apply access controls to video paths and index manifests.

## Limitations
CLIP may miss motion, negation, small objects, long-range temporal relations, audio, and precise counts. Uniform 2-second sampling can miss brief events. No audio understanding or video question-answering is claimed. A complete enterprise system additionally needs permissions filtering, encrypted storage, job queues, index versioning, incremental updates, and domain-specific evaluation.
