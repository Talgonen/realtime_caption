# RTVTLM Project Roadmap

This document outlines the detailed roadmap for the Real-Time Vision-Language Model project, from current state to full background assistant implementation.

---

## 🎯 Project Stages Overview

| Stage | Goal | Status | Timeline |
|-------|------|--------|----------|
| **Stage 1** | CLIP-Qwen Adapter Training | ✅ Complete | Complete |
| **Stage 2** | Video Understanding | 🔄 In Progress | 6-8 weeks |
| **Stage 3** | Background Life Assistant | 🔮 Planned | 8-10 weeks |

---

## Stage 1: ✅ Image Captioning (COMPLETE)

### Achievements
- [x] Hybrid CLIP-Qwen architecture with mid-layer injection
- [x] Cross-attention adapters at layers [15, 17, 19, 21, 23]
- [x] PyTorch Lightning training infrastructure
- [x] COCO Karpathy dataset integration (75K samples)
- [x] Mixed precision training (FP16)
- [x] Real-time GUI application
- [x] Inference scripts
- [x] Configuration system (YAML)
- [x] Checkpoint management

### Key Results
- **Trainable Parameters**: ~2-5M (<1% of total)
- **Training Speed**: 60-500 samples/sec (GPU dependent)
- **Memory Efficiency**: 6GB GPU sufficient
- **Inference Latency**: 30-500ms (hardware dependent)

---

## Stage 2: 🔄 Video Understanding (IN PROGRESS)

**Goal**: Enable the model to process video streams and generate attention-aware, temporally-coherent captions.

### Phase 2.1: Video Foundation (Weeks 1-3)

#### 2.1.1 Video Data Pipeline
- [ ] **Create `dataset/Video.py`**
  - Frame extraction from video files
  - Uniform sampling (N frames per video)
  - Key-frame detection (scene change)
  - Frame buffering system (deque/queue)
  - Support formats: MP4, AVI, MOV, WEBM

```python
class VideoDataset:
    - Extract frames at fixed FPS (e.g., 1 FPS)
    - Or extract on scene change (opencv scene detection)
    - Return: List[PIL.Image], timestamps
```

- [ ] **Integrate video datasets**
  - MSR-VTT (10K videos, 200K captions) - Primary target
  - YouCook2 (2K cooking videos) - Optional
  - ActivityNet Captions (20K videos) - Optional
  - Create unified loader: `get_video_dataset(name='msr-vtt')`

#### 2.1.2 Temporal Modeling
- [ ] **Extend `model/RTVTLM.py` → `model/VideoRTVTLM.py`**
  - Add temporal positional embeddings
  - Frame feature aggregation strategies:
    - **Option A**: Concatenate all frames `[f1, f2, ..., fn]`
    - **Option B**: Temporal attention (weighted average)
    - **Option C**: Recurrent aggregation (LSTM/GRU)

```python
class VideoRTVTLM(RTVTLMForCausalLM):
    def __init__(self):
        super().__init__()
        self.temporal_embeddings = nn.Embedding(max_frames, hidden_size)
        self.frame_aggregator = TemporalAttention()

    def encode_video(self, frames: List[Tensor]) -> Tensor:
        # Batch encode all frames through CLIP
        frame_features = [self.encode_vision(f) for f in frames]
        # Add temporal position
        # Aggregate across time
        return aggregated_features
```

- [ ] **Training strategy**
  - Start with 4-8 frames per video
  - Uniform sampling initially
  - Fine-tune Stage 1 checkpoint on video data
  - Keep adapters trainable, freeze CLIP/Qwen

#### 2.1.3 Testing & Validation
- [ ] Test on short video clips (5-10 seconds)
- [ ] Evaluate temporal coherence
- [ ] Measure frame-to-frame consistency
- [ ] Benchmark inference speed (frames/sec)

**Deliverable**: Model can caption short video clips with temporal understanding.

---

### Phase 2.2: Attention-Aware Generation (Weeks 4-6)

#### 2.2.1 Attention Tracking System
- [ ] **Create `utils/attention_tracker.py`**
  - **Option A**: Mouse position tracking (simple, works immediately)
  - **Option B**: Eye-gaze tracking (requires hardware)
  - Coordinate normalization (0-1 range)
  - Smoothing filter (moving average)

```python
class AttentionTracker:
    def get_focus_point(self) -> Tuple[float, float]:
        # Returns (x, y) in normalized coords [0, 1]
        pass

    def is_dwelling(self, threshold_seconds=2.0) -> bool:
        # Returns True if focus point hasn't moved significantly
        pass
```

#### 2.2.2 Object Detection Integration
- [ ] **Add object detection**: `utils/object_detector.py`
  - Integrate YOLO or DINO
  - Detect objects in each frame
  - Track object IDs across frames
  - Match attention point to detected objects

```python
class ObjectDetector:
    def detect(self, frame: Image) -> List[Detection]:
        # Returns: [Detection(bbox, label, confidence, id), ...]
        pass

    def track(self, detections: List[Detection], prev_frame_detections):
        # Assign consistent IDs across frames
        pass

    def get_focused_object(self, detections, attention_point):
        # Returns object that attention point is on
        pass
```

#### 2.2.3 ROI-Based Detail Generation
- [ ] **Create `model/AttentionAwareVideoRTVTLM.py`**
  - Detect which object user is looking at
  - If dwelling → generate more tokens about that object
  - If moving → general scene description
  - Smooth transitions between objects

```python
class AttentionAwareGenerator:
    def __init__(self):
        self.base_model = VideoRTVTLM()
        self.object_detector = ObjectDetector()
        self.attention_tracker = AttentionTracker()
        self.dwell_history = {}

    def generate_streaming(self, frame, prev_generations):
        # 1. Get attention point
        focus_point = self.attention_tracker.get_focus_point()

        # 2. Detect objects
        objects = self.object_detector.detect(frame)

        # 3. Find focused object
        focused_obj = self.object_detector.get_focused_object(
            objects, focus_point
        )

        # 4. Generate based on dwell time
        if self.attention_tracker.is_dwelling():
            # Generate more detail about focused_obj
            prompt = f"Describe the {focused_obj.label} in detail: "
            return self.base_model.generate(prompt, max_new_tokens=10)
        else:
            # General description
            return self.base_model.generate("Scene: ", max_new_tokens=5)
```

#### 2.2.4 Streaming Token Generation
- [ ] Implement streaming inference
  - Generate tokens continuously as video plays
  - Buffer mechanism for smooth output
  - Handle scene changes (reset generation)
  - Maintain context across frames

**Deliverable**: Model generates detailed descriptions of objects when user dwells on them.

---

### Phase 2.3: Real-Time Video Application (Weeks 7-8)

#### 2.3.1 Video Capture & Processing
- [ ] **Update `main.py` for video**
  - Replace static screenshots with video stream
  - Frame buffer (circular buffer, 10 frames)
  - Async processing pipeline
  - Display streaming captions

```python
class VideoCaptionOverlay:
    def __init__(self):
        self.frame_buffer = deque(maxlen=10)
        self.caption_buffer = []
        self.model = AttentionAwareVideoRTVTLM()

    def capture_loop(self):
        while True:
            frame = capture_screen_frame()
            self.frame_buffer.append(frame)

            # Process every N frames
            if len(self.frame_buffer) == 10:
                tokens = self.model.generate_streaming(
                    list(self.frame_buffer),
                    self.caption_buffer
                )
                self.update_display(tokens)
```

#### 2.3.2 Performance Optimization
- [ ] Parallel frame encoding (batch CLIP inference)
- [ ] Model quantization (INT8) for faster inference
- [ ] Frame skipping (process every 2-3 frames)
- [ ] GPU pipeline optimization
- [ ] Target: 10+ FPS processing speed

#### 2.3.3 Testing & Polish
- [ ] Test on various video types (screen, webcam, files)
- [ ] Measure latency end-to-end
- [ ] User testing for attention tracking accuracy
- [ ] UI/UX improvements

**Deliverable**: Real-time video captioning application with attention-aware detail generation.

---

### Stage 2 Success Metrics

| Metric | Target |
|--------|--------|
| **Video Caption Quality** | Human eval score > 3.5/5 |
| **Temporal Coherence** | Consistency score > 0.7 |
| **Attention Accuracy** | Focused object detected > 80% |
| **Dwell Detection** | True positive rate > 85% |
| **Processing Speed** | 10+ FPS on RTX 3060 |
| **Latency** | < 200ms end-to-end |

---

## Stage 3: 🔮 Background Life Assistant (PLANNED)

**Goal**: Create an intelligent background assistant that continuously monitors, remembers, and retrieves information about daily activities.

### Phase 3.1: Memory Storage System (Weeks 1-3)

#### 3.1.1 Capture Service
- [ ] **Create `services/capture_service.py`**
  - Background daemon (systemd/launchd)
  - Capture screen every N seconds (configurable: 5-30s)
  - Process through video model
  - Generate observations

```python
class CaptureService:
    def __init__(self, interval=10):
        self.model = VideoRTVTLM()
        self.memory_manager = MemoryManager()
        self.interval = interval

    def run(self):
        while True:
            frame = capture_screen()
            description = self.model.generate_caption(frame)

            self.memory_manager.store_observation(
                timestamp=time.time(),
                description=description,
                metadata={'source': 'screen'}
            )

            time.sleep(self.interval)
```

#### 3.1.2 Storage Backend
- [ ] **Create `memory/storage.py`**
  - JSONL for raw logs (one observation per line)
  - Structured format with timestamps
  - Automatic log rotation (daily files)
  - Compression for old logs

```python
# ~/.assistant/raw/2026-03-18.jsonl
{"timestamp": "2026-03-18T10:30:45Z", "description": "User editing code in VSCode", ...}
{"timestamp": "2026-03-18T10:31:00Z", "description": "Switched to browser, reading documentation", ...}
```

#### 3.1.3 Vector Database Integration
- [ ] **Create `memory/embeddings.py`**
  - Embed each observation (sentence-transformers)
  - Store in vector DB (ChromaDB, Qdrant, or Weaviate)
  - Index by: timestamp, activity type, keywords
  - Metadata filtering support

```python
class EmbeddingStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_db = ChromaDB()

    def store(self, observation: Dict):
        embedding = self.embedding_model.encode(observation['description'])
        self.vector_db.add(
            embedding=embedding,
            metadata={
                'timestamp': observation['timestamp'],
                'description': observation['description']
            }
        )
```

**Deliverable**: Background service storing timestamped observations to disk and vector DB.

---

### Phase 3.2: Privacy & Security (Weeks 4-5)

#### 3.2.1 Sensitive Data Filtering
- [ ] **Create `privacy/filters.py`**
  - Password detection (regex patterns)
  - Credit card number filtering
  - Email/phone masking
  - PII detection (names, addresses)
  - Configurable filter rules

```python
class PrivacyFilter:
    def filter(self, text: str) -> str:
        text = self.remove_passwords(text)
        text = self.mask_credit_cards(text)
        text = self.mask_emails(text)
        return text
```

#### 3.2.2 Local-Only Architecture
- [ ] All data stored locally (no cloud)
- [ ] Encryption at rest (LUKS/FileVault)
- [ ] Optional: Encrypt sensitive observations
- [ ] User consent flow on first run
- [ ] Easy data deletion (GDPR-style)

#### 3.2.3 Transparency & Control
- [ ] UI to view all stored observations
- [ ] Filter/delete by date range or keyword
- [ ] Pause/resume capture
- [ ] Exclusion rules (don't capture specific apps)

**Deliverable**: Privacy-preserving capture system with user control.

---

### Phase 3.3: Retrieval & Query Interface (Weeks 6-8)

#### 3.3.1 Semantic Search
- [ ] **Create `memory/retrieval.py`**
  - Embed user query
  - Vector similarity search
  - Time-based filtering
  - Relevance ranking

```python
class MemoryRetriever:
    def search(self, query: str, time_range=None, top_k=10):
        query_embedding = self.embed(query)

        results = self.vector_db.search(
            query_embedding,
            filter={'timestamp': time_range} if time_range else None,
            top_k=top_k
        )

        return self.rank_by_relevance(results, query)
```

#### 3.3.2 Assistant LLM Integration
- [ ] **Create `assistant/query_handler.py`**
  - Integrate GPT-4/Claude/Qwen for answering
  - Build context from retrieved memories
  - Generate natural language responses
  - Cite sources (timestamps)

```python
class AssistantQuery:
    def answer(self, question: str) -> str:
        # 1. Retrieve relevant memories
        memories = self.retriever.search(question, top_k=10)

        # 2. Build context
        context = self.build_context(memories)

        # 3. Generate answer
        prompt = f"""
        Based on these observations from the user's activity:

        {context}

        Question: {question}

        Provide a helpful answer with timestamps.
        """

        return self.llm.generate(prompt)
```

#### 3.3.3 Query Interface (CLI/GUI)
- [ ] CLI tool: `assistant query "What was I working on yesterday?"`
- [ ] GUI: Tkinter/Electron app
- [ ] Voice input (optional)
- [ ] Natural language queries

**Deliverable**: Working assistant that can answer questions about past activities.

---

### Phase 3.4: Memory Summarization (Weeks 9-10)

#### 3.4.1 Daily Summaries
- [ ] **Create `memory/summarizer.py`**
  - Summarize each day's observations
  - Extract key activities
  - Generate daily report
  - Store summaries separately

```python
class MemorySummarizer:
    def summarize_day(self, date: str):
        observations = self.storage.get_observations(date)

        # Use LLM to summarize
        summary = self.llm.generate(f"""
        Summarize these daily activities in 3-5 bullet points:
        {observations}
        """)

        return summary
```

#### 3.4.2 Long-Term Memory Compression
- [ ] Weekly summaries (compress daily summaries)
- [ ] Monthly summaries (compress weekly summaries)
- [ ] Delete raw observations after N days (keep summaries)
- [ ] Configurable retention policy

#### 3.4.3 Knowledge Graph (Optional)
- [ ] Extract entities (people, projects, tools)
- [ ] Build relationship graph
- [ ] Enable graph-based queries
- [ ] Visualize activity patterns

**Deliverable**: Efficient long-term memory with hierarchical summarization.

---

### Stage 3 Success Metrics

| Metric | Target |
|--------|--------|
| **Capture Reliability** | 99%+ uptime |
| **Storage Efficiency** | < 100MB per day |
| **Query Accuracy** | > 85% relevant results |
| **Query Latency** | < 2 seconds |
| **Privacy Leaks** | 0 (100% local) |
| **User Satisfaction** | > 4/5 rating |

---

## Dependencies & Requirements

### Stage 2 Additional Dependencies
```toml
# Video processing
opencv-python = ">=4.9.0"
av = ">=11.0.0"

# Object detection
ultralytics = ">=8.1.0"  # YOLO
transformers = ">=4.37.0"  # DINO

# Attention tracking
pyautogui = ">=0.9.54"  # Mouse tracking
# OR eye-gaze tracking (requires hardware)
```

### Stage 3 Additional Dependencies
```toml
# Vector database
chromadb = ">=0.4.22"
# OR qdrant-client = ">=1.7.0"

# Embeddings
sentence-transformers = ">=2.3.1"

# Privacy
presidio-analyzer = ">=2.2.0"  # PII detection
presidio-anonymizer = ">=2.2.0"

# LLM integration (choose one)
openai = ">=1.10.0"
anthropic = ">=0.18.0"
# OR use local Qwen
```

---

## Estimated Timeline

| Stage | Duration | End Date |
|-------|----------|----------|
| **Stage 1** | Complete | ✅ Mar 18, 2026 |
| **Stage 2.1** | 3 weeks | Apr 8, 2026 |
| **Stage 2.2** | 3 weeks | Apr 29, 2026 |
| **Stage 2.3** | 2 weeks | May 13, 2026 |
| **Stage 3.1** | 3 weeks | Jun 3, 2026 |
| **Stage 3.2** | 2 weeks | Jun 17, 2026 |
| **Stage 3.3** | 3 weeks | Jul 8, 2026 |
| **Stage 3.4** | 2 weeks | Jul 22, 2026 |
| **Total** | ~18 weeks | Jul 22, 2026 |

---

## Risk & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Video quality degradation** | High | Medium | Test on diverse videos, add quality metrics |
| **Real-time performance** | High | Medium | Optimize with quantization, frame skipping |
| **Privacy concerns** | Critical | Low | Build privacy-first, local-only, user control |
| **Storage explosion** | Medium | High | Implement compression, summarization early |
| **Query accuracy poor** | High | Medium | Use high-quality embeddings, LLM for reranking |

---

## Open Questions

1. **Stage 2**: Which video dataset is best for our use case?
2. **Stage 2**: Hardware or software attention tracking?
3. **Stage 3**: Which vector DB offers best local-first experience?
4. **Stage 3**: Self-hosted LLM (Qwen) vs API (GPT-4)?
5. **Stage 3**: How to handle multi-monitor setups?

---

Last Updated: March 18, 2026
