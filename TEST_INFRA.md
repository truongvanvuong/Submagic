# TEST INFRASTRUCTURE SPECIFICATION (`TEST_INFRA.md`)
**Submagic: Hardsub OCR Accuracy & Timing Synchronization Optimization**

## 1. Overview & Testing Philosophy
The Submagic testing infrastructure is built upon an opaque-box, requirement-driven 4-tier test case methodology. It evaluates the hardsub video processing pipeline running exclusively on CPU architectures, enforcing frame-accurate timestamp synchronization, high-fidelity CJK and Vietnamese diacritics recognition, and robust multi-tier fallback mechanisms.

### Core Principles
1. **Requirement-Driven & Opaque-Box**: Tests are derived authoritatively from `ORIGINAL_REQUEST.md` and `PROJECT.md`, verifying observable outputs (SRT timecodes, edit distances, token counts, HTTP status codes, and SQLite persistence).
2. **Progressive Testability**: Tests validate completed milestone features and dependencies without requiring unbuilt future components to pass.
3. **Deterministic & Self-Contained**: Every test sets up its own isolated temporary database (`aiosqlite` WAL mode) and cleans up temporary media artifacts.
4. **Zero-Flakiness CPU Execution**: Heavy ML models (VietOCR, cloud Gemini API) are deterministically mocked in unit runs while maintaining real algorithmic execution of OpenCV visual filters, RapidFuzz deduplication, and PyAV packet parsing.

---

## 2. Test Execution Environment & Runners

### Environment Requirements
- **Python Runtime**: Python 3.11.18 (`backend/.venv/Scripts/python.exe`)
- **Pytest Framework**: Pytest 9.2.0, `pytest-asyncio` 1.4.1 (mode: `auto`), `anyio` 4.15.1
- **Working Directory**: `d:\Workspace\Submagic`
- **Encoding**: UTF-8 (`PYTHONIOENCODING=utf-8`)

### Test Execution Commands
```powershell
# 1. Run the dedicated 4-tier E2E OCR accuracy and timing test suite:
& "d:\Workspace\Submagic\backend\.venv\Scripts\python.exe" -m pytest backend/tests/test_e2e_ocr_accuracy_and_timing.py -v

# 2. Run the complete backend test suite (all 190 tests across all milestones):
& "d:\Workspace\Submagic\backend\.venv\Scripts\python.exe" -m pytest backend/tests -q

# 3. Run reference evaluation oracle on sample video SRT files:
$env:PYTHONIOENCODING="utf-8"
& "d:\Workspace\Submagic\backend\.venv\Scripts\python.exe" scripts/eval_srt.py Sampales/sub_chuan.srt Sampales/sub_app_xuat.srt
```

---

## 3. 4-Tier Test Taxonomy & Coverage Matrix

| Tier | Category | Purpose | Test File | Cases |
|---|---|---|---|---|
| **Tier 1** | **Feature Coverage (Isolation)** | Validate each architectural feature and contract in strict isolation | `test_e2e_ocr_accuracy_and_timing.py` | 11 |
| **Tier 2** | **Boundary & Corner Cases** | Stress extreme values, zero/negative offsets, empty strings, and adversarial inputs | `test_e2e_ocr_accuracy_and_timing.py` | 6 |
| **Tier 3** | **Cross-Feature Interactions** | Verify multi-feature composition, pipelining, and data consistency across module boundaries | `test_e2e_ocr_accuracy_and_timing.py` | 4 |
| **Tier 4** | **Real-World Scenarios** | Execute against actual media files (`tieu_ngao_giang_ho.mp4`) and ground truth reference oracles | `test_e2e_ocr_accuracy_and_timing.py` | 3 |

### Detailed Tier 1: Feature Coverage (Feature Isolation)
1. `test_tier1_rapidocr_configuration_parameters`:
   - Validates configurable `rapidocr_text_score`, `rapidocr_box_thresh`, `rapidocr_thresh`, `rapidocr_unclip_ratio` in `OCRJobRequest`.
   - Verifies kwargs forwarding into RapidOCR without touching external library code.
2. `test_tier1_roi_multi_axis_padding_and_dấu_nặng`:
   - Validates `pad_left_ratio`, `pad_right_ratio`, and `pad_bottom_ratio` in `crop_and_mask_roi` and `VideoROIStream`.
   - Guarantees bottom padding expands downward to prevent clipping Vietnamese dấu nặng (bottom dot).
3. `test_tier1_vietocr_fallback_warning_transparency`:
   - Validates `resolve_ocr_engine("auto", "vi")` resolving to `vietocr`.
   - Verifies explicit warning log and fallback to RapidOCR when VietOCR weights/dependencies are uninstalled.
4. `test_tier1_pts_extraction_and_monotonicity`:
   - Validates PyAV packet presentation timestamp demuxing.
   - Enforces non-negative timestamps and strictly monotonic ordering in presentation time.
5. `test_tier1_mask_coverage_horizontal_overflow`:
   - Compares horizontal bounding extent of text binary mask (`extract_text_binary_mask`) vs OCR detected lines.
   - Triggers overflow alert when mask extends to the left by >= 80% line height.
6. `test_tier1_adaptive_roi_thresholding_and_stroke_density`:
   - Tests Otsu and percentile luminance thresholding on high-brightness ROI scenes.
   - Enforces acceptance of stroke density in expanded range `[0.001, 0.85]`.
7. `test_tier1_cjk_deduplication_character_level`:
   - Verifies that CJK text comparisons do not rely on whole-word whitespace tokenization.
   - Tests character-level difference threshold preventing false merges.
8. `test_tier1_opencc_t2s_normalization`:
   - Validates conversion of Traditional Chinese subtitles into Simplified Chinese.
   - Tests target phrase "誰能想到東方不敗也敗在別人手下了" -> "谁能想到东方不败也败在别人手下了".
9. `test_tier1_dual_version_subtitle_storage_and_script_counter`:
   - Tests storing both `original_text` and `normalized_text` in `SubtitleSegment`.
   - Tests detection of mixed Traditional and Simplified glyphs and automatic low-confidence tagging.
10. `test_tier1_translation_cache_5tuple_key`:
    - Tests 5-tuple composite key `(source_text, target_lang, context_type, model_name, prompt_version)`.
    - Verifies distinct model names or prompt versions generate isolated cache entries.
11. `test_tier1_selective_gemini_ocr_and_cost_estimator`:
    - Tests opt-in Gemini Vision OCR routing (default off, capped by max lines).
    - Validates token estimation and USD cost computation before activation.

### Detailed Tier 2: Boundary & Corner Cases
1. `test_tier2_timestamp_zero_negative_and_nonzero_stream_start`:
   - Tests video streams with non-zero initial `start_time` (normalized to playback zero).
   - Tests negative offset clamping and boundary 0.0s timestamps.
2. `test_tier2_empty_and_single_char_cjk_handling`:
   - Tests empty string `""`, whitespace `"   "`, and single-character CJK `"好"`, `"谁"`.
   - Enforces 100% exact matching constraint on short CJK subtitles.
3. `test_tier2_extreme_roi_padding_boundaries`:
   - Tests negative padding clamped safely to 0.0.
   - Tests excessive padding clamped to safe maximum (<= 0.50).
   - Prevents array slice errors on full-frame border ROIs (`x=0, y=0, w=1.0, h=1.0`).
4. `test_tier2_low_contrast_and_high_density_scenes`:
   - Evaluates minimal contrast frames (text luminance 190, background 180).
   - Tests full-frame flash suppression via connected component area limits.
5. `test_tier2_cjk_seven_characters_differ_by_single_character`:
   - Adversarial test case from R3:
     `cjk_a = "日月神教大教主"` (7 characters)
     `cjk_b = "日月神教副教主"` (7 characters, differs only by "大" vs "副")
   - Enforces that distinct Chinese sentences are NOT merged despite high Levenshtein ratio (85.7%).
6. `test_tier2_vietnamese_tone_marks_and_diacritics_preservation`:
   - Comprehensive test of all Vietnamese vowels with dấu nặng (`ạ, ặ, ệ, ị, ọ, ộ, ợ, ụ, ự, ỵ`).
   - Verifies Unicode NFC normalization and diacritic retention in text cleaners.

### Detailed Tier 3: Cross-Feature Interactions
1. `test_tier3_pts_timing_with_cjk_dedup`:
   - Preserves earliest PTS start and latest PTS end across deduplicated adjacent frames.
   - Selects higher-confidence OCR recognition on merge.
2. `test_tier3_roi_padding_with_mask_coverage_overflow_guard`:
   - Demonstrates that multi-axis padding expands ROI bounding box, eliminating horizontal mask overflow.
3. `test_tier3_opencc_t2s_with_translation_cache`:
   - Proves Traditional Chinese input normalized via `t2s` hits translation cache entry created by Simplified Chinese, eliminating redundant LLM API calls.
4. `test_tier3_vietnamese_full_res_crop_with_dual_cer_eval`:
   - Evaluates standard CER vs diacritic-stripped CER.
   - Demonstrates that tone-mark errors are isolated without polluting character substitution metrics.

### Detailed Tier 4: Real-World Scenarios
1. `test_tier4_sample_video_pyav_packet_demuxing`:
   - Executes PyAV demuxer directly on `Sampales/tieu_ngao_giang_ho.mp4`.
   - Validates exactly 4,928 video packets, duration ~164.27s, and < 1.0s CPU demuxing speed.
2. `test_tier4_target_case_shui_character_integrity`:
   - Verifies ground truth line "誰能想到東方不敗也敗在別人手下了" (16 characters).
   - Validates preservation of leading character "誰" (U+8AB0) and conversion to Simplified "谁".
3. `test_tier4_eval_srt_ground_truth_oracle_execution`:
   - Executes `scripts/eval_srt.py` on `Sampales/sub_chuan.srt` and `Sampales/sub_app_xuat.srt`.
   - Validates 30 matched pairs, Recall ~96.8%, Precision ~69.8%, CER ~16.3%, and linear drift (+586ms initial offset, -137ms/min drift).

---

## 4. Test Harnesses & Shared Utilities

### Database Isolation Fixture
Every test module configures an isolated SQLite database using `tmp_path`:
```python
@pytest.fixture(autouse=True)
async def setup_isolated_test_db(tmp_path: Path, monkeypatch):
    db_file = tmp_path / "test_e2e_ocr.sqlite3"
    monkeypatch.setattr(db, "db_path", db_file)
    await db.init_db()
    yield
```

### Mocking Guidelines
- **RapidOCR**: Mock `ocr_engine_cache.get_rapidocr` returning synthetic bounding boxes and text lines to avoid ONNX provider initialization overhead in unit assertions.
- **VietOCR**: Mock `ocr_engine_cache.get_vietocr` with `FakeVietOCRPredictor` returning predicted strings and probabilities.
- **Gemini Cloud API**: Mock `ocr_engine_cache.recognize_gemini_batch` returning synthetic JSON objects with `index`, `text`, and `confidence`.

---

## 5. QA Role & Bug Escalation Protocol
- The Test Writer modifies **test files and test documentation only**.
- When an implementation defect or specification violation is detected, it is documented with:
  1. Exact file path and line number.
  2. Input data causing the failure.
  3. Observed vs expected output.
  4. Authoritative requirement reference in `ORIGINAL_REQUEST.md` or `PROJECT.md`.
- Escalations are delivered via `handoff.md` and direct coordination message to the dispatching orchestrator.
