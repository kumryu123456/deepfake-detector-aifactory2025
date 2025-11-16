# Tasks: Deepfake Detection AI Competition Model

**Input**: Design documents from `/specs/001-deepfake-detection-competition/`
**Prerequisites**: plan.md (implementation strategy), spec.md (user stories), research.md (SOTA techniques), data-model.md (architecture), contracts/model-interface.md (interfaces)

**Tests**: Test tasks are included as this is a competition submission requiring validation. Tests ensure model correctness, reproducibility, and submission format compliance.

**Organization**: Tasks are organized by implementation phases aligned with user stories. US1 (Model Development) and US2 (Model Submission) are the participant-facing stories we implement. US3-US6 are platform features managed by competition organizers.

---

## Progress Status

**Last Updated**: 2025-11-17

**Completed**: 26 / 80 tasks (32.5%)

**Current Phase**: Phase 3 - User Story 1 (Model Development and Training)

**Recent Milestones**:
- ✅ Phase 1: Setup (T001-T009) - COMPLETE
- ✅ Phase 2: Foundational Infrastructure - Core complete (T013-T014, T016-T024)
- ✅ Phase 3: Model Architecture (T034-T037) - COMPLETE
- ✅ Phase 3: Training Infrastructure (T038, T040-T041) - COMPLETE

**Next Steps**:
- T039: Implement ModelLoader for inference
- T042-T049: Model training and validation
- T050-T068: Inference pipeline and submission preparation (Phase 4)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2)
- Include exact file paths in descriptions

## Path Conventions

Single project structure (PyTorch deep learning):
- `src/` for source code
- `tests/` for test suites
- `configs/` for configuration files
- `scripts/` for executable scripts
- `notebooks/` for Jupyter notebooks

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Create project structure and install dependencies

- [x] T001 Create project directory structure per plan.md (src/, configs/, scripts/, notebooks/, tests/, data/, checkpoints/, logs/) ✅
- [x] T002 Create requirements.txt with core dependencies (torch==1.13.1+cu118, torchvision==0.14.1+cu118, timm==0.9.12, opencv-python-headless==4.8.1.78, albumentations==1.3.1, pandas==2.1.4, scikit-learn==1.3.2, scipy==1.11.4, numpy==1.24.3, Pillow==10.1.0, facenet-pytorch==2.5.3, pyyaml==6.0.1, tqdm==4.66.1, pytest==7.4.3) ✅
- [x] T003 Create setup.py for package installation ✅
- [x] T004 [P] Create .gitignore for data/, checkpoints/, logs/, __pycache__, *.pyc, .ipynb_checkpoints ✅
- [x] T005 [P] Create README.md with project overview and setup instructions ✅
- [x] T006 [P] Create empty __init__.py files in all src/ subdirectories (src/models/, src/data/, src/training/, src/inference/, src/utils/) ✅
- [x] T007 [P] Create placeholder .gitkeep files in data/, checkpoints/, logs/ directories ✅
- [x] T008 Create configuration templates in configs/ (model_config.yaml, training_config.yaml, inference_config.yaml) ✅
- [x] T009 Initialize git repository and make initial commit ✅

---

## Phase 2: Foundational Infrastructure (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before model development

**⚠️ CRITICAL**: No user story work can begin until this phase is complete. This includes data acquisition, preprocessing infrastructure, and base utilities.

### Data Preparation (Parallel Execution Recommended)

- [ ] T010 [P] Create data download script for FaceForensics++ in scripts/download_faceforensics.sh (reference: https://github.com/ondyari/FaceForensics)
- [ ] T011 [P] Create data download script for DFDC in scripts/download_dfdc.sh (reference: https://ai.facebook.com/datasets/dfdc/)
- [ ] T012 [P] Create data download script for Celeb-DF in scripts/download_celebdf.sh (reference: https://github.com/yuezunli/celeb-deepfakeforensics)
- [x] T013 Implement FaceDetector class in src/data/face_detector.py (RetinaFace backend, detect_faces, crop_face, detect_and_crop methods per contracts/model-interface.md) ✅
- [x] T014 Implement VideoProcessor class in src/data/video_processor.py (extract_frames with uniform sampling, process_video methods per contracts/model-interface.md) ✅
- [ ] T015 Create data preprocessing script in scripts/preprocess_data.py (face detection + cropping for all datasets, save to processed/ subdirectories)

### Core Utilities

- [x] T016 [P] Implement configuration loader in src/utils/config.py (load_config, save_config, merge_configs functions for YAML files) ✅
- [x] T017 [P] Implement logging setup in src/utils/logger.py (setup_logger with file and console handlers, format: timestamp, level, message) ✅

### Data Pipeline

- [x] T018 Implement DeepfakeDataset class in src/data/dataset.py (PyTorch Dataset with __init__, __len__, __getitem__, supports both image and video inputs per contracts/model-interface.md) ✅
- [x] T019 Implement data augmentation transforms in src/data/transforms.py (AlbumentationsTransforms wrapper: horizontal flip, rotation, color jitter, JPEG compression, Gaussian noise/blur per research.md) ✅
- [x] T020 Implement DataPreprocessor class in src/data/transforms.py (__call__ method for normalization, resize to 224x224, tensor conversion per contracts/model-interface.md) ✅

### Metrics and Loss Functions

- [x] T021 [P] Implement MetricsCalculator in src/training/metrics.py (compute_macro_f1, compute_all_metrics static methods per contracts/model-interface.md, print_metrics_report) ✅
- [x] T022 [P] Implement Soft F1 Loss in src/training/losses.py (soft_f1_loss function per research.md lines 270-298, differentiable approximation for binary classification) ✅
- [x] T023 [P] Implement Focal Loss in src/training/losses.py (FocalLoss class with gamma=2.0, alpha=0.25 per research.md) ✅
- [x] T024 Implement CombinedLoss in src/training/losses.py (combines CrossEntropyLoss, FocalLoss, SoftF1Loss with configurable weights per contracts/model-interface.md) ✅

**Checkpoint**: Foundation ready - model development can now begin

---

## Phase 3: User Story 1 - Model Development and Training (Priority: P1) 🎯 MVP

**Goal**: Develop and train a dual-branch deepfake detection model achieving >80% Macro F1-score on cross-dataset validation

**Independent Test**: Train model on FaceForensics++ and DFDC, validate on Celeb-DF, achieve Macro F1 >80% and verify predictions output correct format (0 for Real, 1 for Fake)

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T025 [P] [US1] Unit test for SpatialBranch forward pass in tests/unit/test_models.py (test input shape (4, 3, 224, 224) → output shape (4, 512))
- [ ] T026 [P] [US1] Unit test for FrequencyBranch forward pass in tests/unit/test_models.py (test FFT computation, output shape (4, 512))
- [ ] T027 [P] [US1] Unit test for FusionLayer in tests/unit/test_models.py (test concatenation + attention, input (4, 512) + (4, 512) → output (4, 512))
- [ ] T028 [P] [US1] Unit test for DeepfakeDetector in tests/unit/test_models.py (test end-to-end forward pass, extract_features method)
- [ ] T029 [P] [US1] Unit test for DataPreprocessor in tests/unit/test_data.py (test normalization, resize, tensor conversion)
- [ ] T030 [P] [US1] Unit test for MetricsCalculator in tests/unit/test_metrics.py (test compute_macro_f1 with known inputs/outputs)
- [ ] T031 [P] [US1] Unit test for loss functions in tests/unit/test_metrics.py (test SoftF1Loss, FocalLoss, CombinedLoss with synthetic data)
- [ ] T032 [US1] Integration test for training pipeline in tests/integration/test_training.py (test train_epoch, validate methods with small dataset)
- [ ] T033 [US1] Integration test for data loading in tests/integration/test_data.py (test DeepfakeDataset with real files, DataLoader batching)

### Model Architecture Implementation

- [x] T034 [P] [US1] Implement SpatialBranch in src/models/spatial_branch.py (EfficientNet-B4 backbone + Vision Transformer encoder per data-model.md lines 134-176, output 512-dim features) ✅
- [x] T035 [P] [US1] Implement FrequencyBranch in src/models/frequency_branch.py (FFT transformation + CNN processing per data-model.md lines 178-218, output 512-dim features, handle amplitude and phase spectra) ✅
- [x] T036 [US1] Implement FusionLayer in src/models/fusion_layer.py (hierarchical cross-modal attention per data-model.md lines 220-276, input 1024-dim → output 512-dim) ✅
- [x] T037 [US1] Implement DeepfakeDetector main model in src/models/deepfake_detector.py (integrate SpatialBranch + FrequencyBranch + FusionLayer, forward, predict, extract_features methods per contracts/model-interface.md) ✅

### Training Infrastructure

- [x] T038 [US1] Implement Trainer class in src/training/trainer.py (train, train_epoch, validate, save_checkpoint methods per contracts/model-interface.md, support mixed precision, early stopping on Macro F1) ✅
- [ ] T039 [US1] Implement ModelLoader in src/inference/model_loader.py (load_checkpoint, load_config static methods per contracts/model-interface.md)
- [x] T040 [US1] Create training script in scripts/train.py (parse args, load config, create data loaders, instantiate model, run training loop, save checkpoints) ✅
- [x] T041 [US1] Create evaluation script in scripts/evaluate.py (load checkpoint, run inference on validation set, compute and print all metrics) ✅

### Model Training and Validation

- [ ] T042 [US1] Create baseline training config in configs/baseline_config.yaml (EfficientNet-B4 only, 100 epochs, batch size 32, lr 1e-4, AdamW optimizer, cosine annealing scheduler)
- [ ] T043 [US1] Train baseline model on FaceForensics++ using scripts/train.py with baseline_config.yaml (target: >85% accuracy, save best checkpoint to checkpoints/baseline_best.pth)
- [ ] T044 [US1] Validate baseline model on FaceForensics++ test set using scripts/evaluate.py (verify Macro F1 >80%, per-class precision/recall balanced)
- [ ] T045 [US1] Create hybrid training config in configs/hybrid_config.yaml (dual-branch architecture, combined loss with weights [0.5 CE, 0.3 Focal, 0.2 F1], same training params as baseline)
- [ ] T046 [US1] Train hybrid model on FaceForensics++ + DFDC using scripts/train.py with hybrid_config.yaml (multi-dataset training, balanced sampling, target: >90% internal F1, save to checkpoints/hybrid_best.pth)
- [ ] T047 [US1] Cross-dataset validation: evaluate hybrid model on Celeb-DF using scripts/evaluate.py (target: >80% Macro F1, verify generalization, document results in logs/)
- [ ] T048 [US1] Fine-tune hybrid model with F1-optimized loss schedule (increase F1 loss weight to 0.4, reduce CE to 0.4, Focal 0.2, fine-tune for 10 epochs, save to checkpoints/hybrid_finetuned.pth)
- [ ] T049 [US1] Run compression robustness test: evaluate on JPEG-compressed Celeb-DF (quality 70-90) using scripts/evaluate.py (verify performance degradation <5%)

**Checkpoint**: Model trained and validated independently, achieves target Macro F1 >80% on cross-dataset evaluation

---

## Phase 4: User Story 2 - Model Submission and Automated Evaluation (Priority: P1)

**Goal**: Package trained model into task.ipynb for competition submission, ensure inference completes within 3 hours, output correct submission.csv format

**Independent Test**: Run task.ipynb locally on sample data, verify submission.csv generated with correct format, validate all files processed within time limit

### Tests for User Story 2

- [ ] T050 [P] [US2] Unit test for InferenceEngine.process_image in tests/unit/test_inference.py (test single image inference, output label in {0, 1})
- [ ] T051 [P] [US2] Unit test for InferenceEngine.process_video in tests/unit/test_inference.py (test video frame extraction + aggregation, single label output)
- [ ] T052 [P] [US2] Unit test for submission CSV validation in tests/unit/test_inference.py (test validate_submission_format from quickstart.md lines 650-687)
- [ ] T053 [US2] Integration test for full inference pipeline in tests/integration/test_inference.py (test run_inference on sample data directory, verify submission.csv format and completeness)
- [ ] T054 [US2] Contract test for submission format in tests/contract/test_submission.py (test CSV has exactly 2 columns [filename, label], all labels in {0, 1}, all filenames have extensions, no null values per quickstart.md)

### Inference Pipeline Implementation

- [ ] T055 [US2] Implement InferenceEngine in src/inference/inference_engine.py (run_inference, process_image, process_video, aggregate_frame_predictions methods per contracts/model-interface.md, support batch processing, FP16, video frame sampling)
- [ ] T056 [US2] Create inference script in scripts/inference.py (load checkpoint, create InferenceEngine, run inference on ./data/, save to submission.csv, print summary statistics)
- [ ] T057 [US2] Create submission validation script in scripts/test_submission.py (implement validate_and_fix_submission, create_submission_with_validation per quickstart.md lines 689-841)

### Submission Notebook Creation

- [ ] T058 [US2] Create task.ipynb in notebooks/ (Cell 1: pip install all dependencies with exact versions, Cell 2: import all modules, Cell 3: load model checkpoint from ./checkpoints/, Cell 4: initialize InferenceEngine)
- [ ] T059 [US2] Add inference execution to task.ipynb (Cell 5: run inference on ./data/ directory, Cell 6: save results to submission.csv, Cell 7: validate submission format, Cell 8: print summary with Real/Fake counts)
- [ ] T060 [US2] Add error handling to task.ipynb (try-except blocks for file I/O, model loading, inference, CSV writing, print error traceback on failure)
- [ ] T061 [US2] Test task.ipynb locally with competition sample data (download 7 fake images + 5 fake videos per spec.md FR-001, run notebook end-to-end, verify submission.csv generated correctly)
- [ ] T062 [US2] Benchmark inference time on simulated test set (~10,000 samples) (create mock dataset with mixed images/videos, measure total inference time, verify <2 hours for safety margin per plan.md)
- [ ] T063 [US2] Optimize inference for speed (enable FP16, adjust batch size to 64, reduce video frames to 16 if needed, implement fallback for OOM per plan.md lines 325-401)
- [ ] T064 [US2] Test reproducibility: run task.ipynb 3 times (fix random seeds in notebook, verify submission.csv identical across runs, document seed values)

### Submission Package Preparation

- [ ] T065 [US2] Package model checkpoint in notebooks/checkpoints/ (copy best hybrid model, ensure file size <500MB if possible, document model architecture in README)
- [ ] T066 [US2] Add preprocessing documentation to task.ipynb (document face detection settings, image resize dimensions, normalization values, frame sampling strategy in markdown cells)
- [ ] T067 [US2] Create submission checklist in task.ipynb (markdown cell: verify dependencies installed, checkpoint loaded, inference runs, submission.csv validated, reproducibility tested)
- [ ] T068 [US2] Final validation: run complete submission simulation (clean environment, execute task.ipynb from scratch, verify all outputs correct, time execution)

**Checkpoint**: Submission package (task.ipynb + checkpoints) ready for competition upload, all validations passed

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Final improvements and documentation for competition submission

- [ ] T069 [P] Update README.md with complete setup instructions (environment setup, data download, training commands, inference commands, submission preparation)
- [ ] T070 [P] Create EDA notebook in notebooks/eda.ipynb (visualize dataset statistics, class distribution, sample images/videos, face detection results)
- [ ] T071 [P] Create experiments tracking notebook in notebooks/experiments.ipynb (log training runs, compare baseline vs hybrid, document hyperparameter tuning, visualize learning curves)
- [ ] T072 Document model architecture in docs/architecture.md (create docs/ directory, describe dual-branch design, include architecture diagram from data-model.md, explain design decisions)
- [ ] T073 Document training procedure in docs/training.md (describe dataset preparation, augmentation strategy, loss function schedule, validation protocol, checkpoint selection criteria)
- [ ] T074 [P] Add code comments and docstrings to all src/ modules (ensure all classes and functions have docstrings per contracts/model-interface.md section 10, add type hints)
- [ ] T075 Run full test suite with pytest (execute pytest tests/ -v, ensure all tests pass, aim for >80% code coverage on critical modules)
- [ ] T076 Profile memory usage during inference (use torch.cuda.max_memory_allocated(), verify peak usage <20GB for L4 GPU per plan.md OOM mitigation)
- [ ] T077 [P] Code cleanup and refactoring (remove debug print statements, unused imports, commented-out code, ensure PEP 8 compliance)
- [ ] T078 Create quickstart validation script in scripts/validate_quickstart.py (verify all steps in quickstart.md work, test sample data download, training command, inference command, submission validation)
- [ ] T079 Final submission preparation: compress task.ipynb and checkpoints/ directory (ensure no .git folders, no unnecessary files, verify total size reasonable per spec.md FR-019)
- [ ] T080 Document known limitations and future improvements in docs/limitations.md (generalization gap notes per research.md, compression robustness, OOM edge cases, potential optimizations)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup (Phase 1) completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational (Phase 2) completion - Core model development
- **User Story 2 (Phase 4)**: Depends on User Story 1 (Phase 3) completion - Requires trained model
- **Polish (Phase 5)**: Depends on User Stories 1 and 2 completion - Final touches

### Critical Path

1. **Setup** (T001-T009) → **Foundation** (T010-T024) → **US1 Tests** (T025-T033) → **US1 Models** (T034-T037) → **US1 Training** (T038-T049) → **US2 Tests** (T050-T054) → **US2 Inference** (T055-T057) → **US2 Submission** (T058-T068) → **Polish** (T069-T080)

### Within Each Phase

**Phase 2 - Foundational**:
- T010, T011, T012 can run in parallel (data downloads)
- T016, T017 can run in parallel (utilities)
- T021, T022, T023 can run in parallel (loss functions)
- T013, T014 must complete before T015 (preprocessing needs face detection and video processing)
- T018, T019, T020 depend on T013, T014

**Phase 3 - User Story 1**:
- T025-T033 (tests) can all run in parallel after writing
- T034, T035 can run in parallel (independent branches)
- T036 depends on T034, T035 (fusion needs both branches)
- T037 depends on T034, T035, T036 (main model integrates all)
- T038, T039, T040, T041 can proceed once T037 is done
- T042-T049 (training) are sequential

**Phase 4 - User Story 2**:
- T050, T051, T052 can run in parallel (unit tests)
- T053, T054 depend on T055 (integration tests need InferenceEngine)
- T055, T056, T057 are sequential
- T058-T068 (notebook development) are mostly sequential

**Phase 5 - Polish**:
- T069, T070, T071, T074, T077 can run in parallel (documentation and cleanup)
- T075, T076, T078 are validation tasks that can run in parallel

### Parallel Opportunities

**Setup Phase**: T004, T005, T006, T007 (4 tasks in parallel)
**Foundational Phase**: T010-T012 (data downloads), T016-T017 (utilities), T021-T023 (losses) = 8 tasks in parallel
**US1 Tests**: T025-T033 (9 tests in parallel)
**US1 Models**: T034, T035 (2 branches in parallel)
**US2 Tests**: T050-T052 (3 tests in parallel)
**Polish Phase**: T069-T071, T074, T077 (5 tasks in parallel)

---

## Parallel Execution Examples

### Parallel Example: Foundational Phase Data Downloads

```bash
# Launch all data download tasks together:
Task: "Create data download script for FaceForensics++ in scripts/download_faceforensics.sh"
Task: "Create data download script for DFDC in scripts/download_dfdc.sh"
Task: "Create data download script for Celeb-DF in scripts/download_celebdf.sh"

# Execute downloads in parallel (background processes):
bash scripts/download_faceforensics.sh &
bash scripts/download_dfdc.sh &
bash scripts/download_celebdf.sh &
wait  # Wait for all to complete
```

### Parallel Example: User Story 1 Tests

```bash
# Launch all unit tests for US1 together:
Task: "Unit test for SpatialBranch in tests/unit/test_models.py"
Task: "Unit test for FrequencyBranch in tests/unit/test_models.py"
Task: "Unit test for FusionLayer in tests/unit/test_models.py"
Task: "Unit test for DeepfakeDetector in tests/unit/test_models.py"
Task: "Unit test for DataPreprocessor in tests/unit/test_data.py"
Task: "Unit test for MetricsCalculator in tests/unit/test_metrics.py"
Task: "Unit test for loss functions in tests/unit/test_metrics.py"

# Run with pytest in parallel mode:
pytest tests/unit/ -n auto  # Uses all CPU cores
```

### Parallel Example: User Story 1 Model Branches

```bash
# Implement spatial and frequency branches in parallel:
Task: "Implement SpatialBranch in src/models/spatial_branch.py"
Task: "Implement FrequencyBranch in src/models/frequency_branch.py"

# Two developers can work simultaneously on different files
```

---

## Implementation Strategy

### MVP First (User Stories 1 and 2)

1. Complete Phase 1: Setup (T001-T009)
2. Complete Phase 2: Foundational (T010-T024) - **CRITICAL checkpoint**
3. Complete Phase 3: User Story 1 (T025-T049) - **Model trained and validated**
4. Complete Phase 4: User Story 2 (T050-T068) - **Submission package ready**
5. **STOP and VALIDATE**: Test complete submission flow locally
6. Submit to competition

### Incremental Delivery Milestones

1. **Milestone 1**: Foundation Ready (after T024)
   - Project structure complete
   - Data pipeline functional
   - Training infrastructure ready
   - Can begin model development

2. **Milestone 2**: Baseline Model Trained (after T044)
   - Working baseline model
   - >85% accuracy on FaceForensics++
   - Validates training pipeline works
   - Can proceed to hybrid architecture

3. **Milestone 3**: Hybrid Model Trained (after T047)
   - Dual-branch model complete
   - >80% cross-dataset Macro F1
   - Validates generalization
   - Can proceed to submission preparation

4. **Milestone 4**: Submission Ready (after T068)
   - task.ipynb complete and tested
   - Inference validated
   - Format compliance verified
   - Ready for competition upload

5. **Milestone 5**: Competition Submission (after T080)
   - All documentation complete
   - Code cleaned and commented
   - Final validation passed
   - Submission uploaded

### Time Estimates (from plan.md)

- **Phase 1 (Setup)**: 1 day
- **Phase 2 (Foundational)**: 1-2 weeks (parallel data downloads can reduce to 1 week)
- **Phase 3 (User Story 1)**: 3-4 weeks (baseline: 2 weeks, hybrid: 1-2 weeks)
- **Phase 4 (User Story 2)**: 1 week
- **Phase 5 (Polish)**: 3-5 days

**Total Estimated Time**: 6 weeks (matches plan.md timeline)

### Risk Mitigation During Implementation

Per plan.md risk analysis:

- **After T024**: Run memory profiling benchmark (OOM risk mitigation)
- **After T044**: Verify baseline accuracy >85% (if not, debug before proceeding)
- **After T047**: Verify cross-dataset F1 >80% (if not, increase augmentation or add more data)
- **After T062**: Verify inference time <2 hours (if not, optimize batch size, reduce frames, or use FP16)
- **After T064**: Verify reproducibility (if fails, check random seeds and deterministic operations)

---

## Notes

- **[P] tasks**: Different files, no dependencies, can run in parallel
- **[Story] labels**: Track tasks to user stories (US1 = Model Development, US2 = Submission)
- **Tests first**: Write and verify tests fail before implementing features (TDD approach)
- **Checkpoints**: Validate at each milestone before proceeding
- **Commit frequency**: After each task or logical group of related tasks
- **User Stories 3-6**: These are competition platform features (team management, CUDA environments, timeline tracking, rules enforcement) implemented by competition organizers, not participants. Our implementation focuses on US1 and US2 only.

---

## Total Task Count

- **Phase 1 (Setup)**: 9 tasks
- **Phase 2 (Foundational)**: 15 tasks
- **Phase 3 (User Story 1)**: 25 tasks (9 tests + 16 implementation)
- **Phase 4 (User Story 2)**: 19 tasks (5 tests + 14 implementation)
- **Phase 5 (Polish)**: 12 tasks
- **TOTAL**: 80 tasks

**Parallel Opportunities**: 28 tasks marked [P] can run in parallel with other tasks

**Test Coverage**: 14 test tasks ensure model correctness, inference reliability, and submission compliance

**MVP Scope**: Phases 1-4 (T001-T068) deliver a complete, competition-ready submission
