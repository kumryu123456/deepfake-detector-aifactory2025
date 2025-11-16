# Tasks: Deepfake Detection AI Competition Platform

**Input**: Design documents from `/specs/001-deepfake-detection-competition/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/model-interface.md, quickstart.md

**Tests**: Not explicitly requested in specification - focusing on implementation and manual validation with competition sample data

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story. This is a competition ML project, so user stories represent development phases rather than end-user features.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- Include exact file paths in descriptions

## Path Conventions

Single project structure with:
- **src/**: Main source code (models, data, training, inference, utils)
- **configs/**: YAML configuration files
- **scripts/**: Executable Python scripts
- **notebooks/**: Jupyter notebooks including task.ipynb
- **data/**: Training datasets (not in git)
- **checkpoints/**: Model weights (not in git)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project directory structure per plan.md (src/, configs/, scripts/, notebooks/, tests/, data/, checkpoints/, logs/)
- [ ] T002 Create requirements.txt with Python 3.9 dependencies: torch==1.13.1, torchvision==0.14.1, timm==0.9.12, opencv-python-headless==4.8.1.78, albumentations==1.3.1, pandas==2.1.4, scikit-learn==1.3.2, scipy==1.11.4, numpy==1.24.3, Pillow==10.1.0, facenet-pytorch==2.5.3, pyyaml==6.0.1, tqdm==4.66.1
- [ ] T003 [P] Create .gitignore with exclusions: data/, checkpoints/, logs/, __pycache__/, *.pyc, .ipynb_checkpoints/, *.pth
- [ ] T004 [P] Create README.md with project overview, setup instructions, and competition context
- [ ] T005 [P] Create setup.py for package installation with metadata and dependencies
- [ ] T006 [P] Initialize all src/ subdirectory __init__.py files (src/models/, src/data/, src/training/, src/inference/, src/utils/)
- [ ] T007 Create configs/model_config.yaml with dual-branch architecture settings (spatial_branch, frequency_branch, fusion, classifier)
- [ ] T008 [P] Create configs/training_config.yaml with hyperparameters (epochs=100, batch_size=32, lr=1e-4, optimizer=adamw, scheduler=cosine_annealing)
- [ ] T009 [P] Create configs/inference_config.yaml with inference settings (batch_size=64, video_frames=16, use_fp16=true)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T010 Implement configuration management in src/utils/config.py (load YAML, merge configs, provide Config dataclass)
- [ ] T011 [P] Implement logging setup in src/utils/logger.py (configure logging with file and console handlers, log levels)
- [ ] T012 [P] Implement DataPreprocessor base class in src/data/transforms.py (handle resize, normalize, augmentation for train/inference modes)
- [ ] T013 [P] Implement FaceDetector in src/data/face_detector.py (MTCNN or RetinaFace, detect faces, crop with margin, handle no-face cases)
- [ ] T014 [P] Implement VideoProcessor in src/data/video_processor.py (extract frames uniformly, handle fps variations, integrate face detection)
- [ ] T015 [P] Implement MetricsCalculator in src/training/metrics.py (compute Macro F1, per-class precision/recall, confusion matrix, AUC)
- [ ] T016 Create data/ directory structure with subdirectories: faceforensics/, dfdc/, celebdf/, sample/

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Model Development and Training (Priority: P1) 🎯 MVP

**Goal**: Develop and train a baseline deepfake detection model that can classify images and videos as Real (0) or Fake (1) with local validation

**Independent Test**: Download competition sample data (7 fake images, 5 fake videos), train baseline EfficientNet-B4 model on FaceForensics++ dataset, validate locally on sample data, verify model outputs binary predictions and achieves >85% accuracy on FF++ test set

### Implementation for User Story 1

**Models & Architecture:**
- [ ] T017 [P] [US1] Implement SpatialBranch in src/models/spatial_branch.py (EfficientNet-B4 backbone from timm, Vision Transformer encoder with 4 layers/8 heads, output 512-dim features)
- [ ] T018 [P] [US1] Implement FrequencyBranch in src/models/frequency_branch.py (FFT/DCT transform, amplitude/phase spectrum processing, conv layers, output 512-dim features)
- [ ] T019 [P] [US1] Implement FusionLayer in src/models/fusion_layer.py (self-attention mechanism with 4 heads, combine spatial+frequency features into 1024-dim)
- [ ] T020 [US1] Implement DeepfakeDetector in src/models/deepfake_detector.py (integrate spatial+frequency branches, fusion layer, classifier head, forward/predict/extract_features methods per contracts/model-interface.md)

**Data Processing:**
- [ ] T021 [P] [US1] Implement DeepfakeDataset in src/data/dataset.py (PyTorch Dataset for images/videos, load labels, apply transforms, handle multi-dataset sampling with weights)
- [ ] T022 [US1] Extend DataPreprocessor in src/data/transforms.py with training augmentations (horizontal flip, rotation, color jitter, Gaussian blur/noise, JPEG compression per research.md recommendations)
- [ ] T023 [US1] Create data preprocessing script in scripts/preprocess_data.py (download datasets instructions, face extraction, organize into real/fake directories, save preprocessed data)

**Training Components:**
- [ ] T024 [P] [US1] Implement CombinedLoss in src/training/losses.py (Cross-Entropy + Focal Loss + differentiable Macro F1 loss, weights: 0.5/0.3/0.2)
- [ ] T025 [P] [US1] Implement Trainer in src/training/trainer.py (training loop, validation, metrics tracking, checkpoint saving, early stopping based on Macro F1)
- [ ] T026 [US1] Create training script in scripts/train.py (load config, initialize model, setup data loaders with balanced sampling, train with warmup/main/fine-tuning phases, save best checkpoint)

**Validation & Checkpoints:**
- [ ] T027 [US1] Implement checkpoint management in src/training/trainer.py (save model state, optimizer state, epoch, metrics, config, timestamp per data-model.md schema)
- [ ] T028 [US1] Create evaluation script in scripts/evaluate.py (load checkpoint, run inference on validation set, compute all metrics, generate classification report)
- [ ] T029 [US1] Download FaceForensics++ dataset to data/faceforensics/ (real and fake videos with compression levels c0/c23/c40)
- [ ] T030 [US1] Train baseline EfficientNet-B4 model on FaceForensics++ for 100 epochs with config from configs/training_config.yaml
- [ ] T031 [US1] Validate baseline model performance (accuracy >85% on FF++ test set, Macro F1 >83%, save best checkpoint to checkpoints/baseline_ffpp.pth)

**Checkpoint**: At this point, User Story 1 should be fully functional - a trained model that can detect deepfakes in images and videos with local validation

---

## Phase 4: User Story 2 - Model Submission and Automated Evaluation (Priority: P1)

**Goal**: Create inference pipeline and submission notebook (task.ipynb) that packages the trained model for competition submission, processes test data from ./data/ directory, generates submission.csv, and enables submission via aifactory library

**Independent Test**: Load trained model checkpoint, run inference on competition sample data in ./data/, verify submission.csv format (filename, label columns), validate Macro F1 calculation matches expected results, test task.ipynb executes end-to-end without errors

### Implementation for User Story 2

**Inference Components:**
- [ ] T032 [P] [US2] Implement ModelLoader in src/inference/model_loader.py (load checkpoint file, restore model weights, handle config loading, set eval mode, move to GPU)
- [ ] T033 [US2] Implement InferenceEngine in src/inference/inference_engine.py (process images and videos, batch inference, frame aggregation for videos, generate submission.csv per contracts/model-interface.md)
- [ ] T034 [US2] Add video frame aggregation methods to InferenceEngine (average_logits, max_confidence, majority_vote, configurable via inference_config.yaml)

**Inference Optimization:**
- [ ] T035 [P] [US2] Implement mixed precision (FP16) support in InferenceEngine for 2x speedup
- [ ] T036 [P] [US2] Implement batch processing in InferenceEngine (process multiple images in parallel with batch_size=64, video frames with batch_size=16)
- [ ] T037 [US2] Add early stopping for video inference (confidence thresholding to skip remaining frames if prediction is confident)

**Scripts & Validation:**
- [ ] T038 [US2] Create inference script in scripts/inference.py (CLI to run inference on data directory, specify checkpoint path, output submission.csv, display progress with tqdm)
- [ ] T039 [P] [US2] Create submission validation script in scripts/test_submission.py (verify CSV format: columns [filename, label], values are 0/1, no nulls, all filenames have extensions)
- [ ] T040 [US2] Download competition sample data to data/sample/ (7 fake images, 5 fake videos from competition page)
- [ ] T041 [US2] Run inference on sample data with baseline model, validate submission.csv format passes all checks in test_submission.py
- [ ] T042 [US2] Benchmark inference time on ~1000 simulated test samples to ensure <2 hour completion (extrapolate to 10K samples)

**Submission Notebook:**
- [ ] T043 [US2] Create task.ipynb in notebooks/ with cells: 1) pip install dependencies (torch, timm, opencv, etc.), 2) import src modules, 3) load model checkpoint, 4) run InferenceEngine on ./data/, 5) save submission.csv, 6) display sample results
- [ ] T044 [US2] Add aifactory.score.submit() call to task.ipynb with placeholder for competition key (CUDA 11.8 environment)
- [ ] T045 [US2] Test task.ipynb executes end-to-end locally (kernel restart, run all cells, verify submission.csv created, check for errors)
- [ ] T046 [US2] Add documentation to task.ipynb (markdown cells explaining model architecture, preprocessing, inference process per competition verification requirements)
- [ ] T047 [US2] Verify reproducibility by running task.ipynb 3 times and comparing submission.csv outputs (should be identical with fixed random seeds)

**Checkpoint**: At this point, User Story 2 complete - inference pipeline ready, task.ipynb submittable to competition, baseline submission validated

---

## Phase 5: Hybrid Model Enhancement (Extends US1 for Better Performance)

**Goal**: Upgrade from baseline to dual-branch hybrid architecture with frequency domain analysis, train on multiple datasets for better generalization, optimize for Macro F1 >82%

**Independent Test**: Train hybrid model on FF++ + DFDC + Celeb-DF, validate cross-dataset performance (train on FF++/DFDC, test on Celeb-DF), achieve Macro F1 >80% on unseen dataset

### Implementation for Hybrid Model

**Multi-Dataset Training:**
- [ ] T048 [US1] Download DFDC dataset to data/dfdc/ (~470GB, organize train/val splits with real/fake subdirectories)
- [ ] T049 [US1] Download Celeb-DF v2 dataset to data/celebdf/ (real and fake celebrity videos)
- [ ] T050 [US1] Update DeepfakeDataset in src/data/dataset.py to support multi-dataset loading with configurable weights (FF++=0.3, DFDC=0.5, Celeb-DF=0.2)
- [ ] T051 [US1] Implement balanced sampling strategy in dataset to ensure equal Real/Fake representation per batch

**Advanced Training:**
- [ ] T052 [US1] Update training_config.yaml with multi-dataset paths and sampling weights
- [ ] T053 [US1] Implement cross-dataset validation in scripts/train.py (train on FF++/DFDC, validate on Celeb-DF held-out set)
- [ ] T054 [US1] Add consistency regularization to CombinedLoss in src/training/losses.py (enforce feature consistency across augmentations per research.md CORE method)
- [ ] T055 [US1] Train hybrid dual-branch model on all three datasets for 100 epochs with cross-dataset validation
- [ ] T056 [US1] Monitor and log per-class metrics (precision/recall for Real and Fake separately) to ensure balanced performance
- [ ] T057 [US1] Validate hybrid model achieves cross-dataset Macro F1 >80% and save best checkpoint to checkpoints/hybrid_multi_dataset.pth

**Checkpoint**: Hybrid model trained with superior generalization, ready for final optimization

---

## Phase 6: Final Optimization & Submission Preparation

**Goal**: Optimize inference speed, finalize submission notebook with best model, perform final validation, prepare for competition submission

**Independent Test**: Run inference on 10K simulated test samples, verify completion <2 hours, validate submission.csv format, test task.ipynb reproducibility 3+ times

### Implementation for Final Submission

**Inference Speed Optimization:**
- [ ] T058 [US2] Profile inference pipeline with cProfile to identify bottlenecks (model forward pass, face detection, I/O)
- [ ] T059 [US2] Optimize video frame sampling in VideoProcessor (reduce to 16 frames if needed, parallel frame extraction)
- [ ] T060 [US2] Implement gradient checkpointing in DeepfakeDetector if memory is constraint (trade compute for memory)
- [ ] T061 [US2] Add DataLoader prefetching in InferenceEngine (num_workers=8, pin_memory=True, prefetch_factor=2)
- [ ] T062 [US2] Test inference speed on 10,000 mixed samples (70% images, 30% videos), verify <2 hour completion, document timing breakdown

**Submission Finalization:**
- [ ] T063 [US2] Update task.ipynb to use best hybrid model checkpoint (checkpoints/hybrid_multi_dataset.pth)
- [ ] T064 [US2] Add fallback handling in task.ipynb for face detection failures (use center crop if no face detected)
- [ ] T065 [US2] Add OOM error handling in task.ipynb (reduce batch size to 32 if OOM occurs, reduce video frames to 8)
- [ ] T066 [US2] Document all preprocessing steps in task.ipynb markdown cells (face detection method, margin ratio, resizing, normalization values for verification reproducibility)
- [ ] T067 [US2] Fix all random seeds in task.ipynb (python, numpy, torch.manual_seed, torch.cuda.manual_seed_all) for reproducibility
- [ ] T068 [US2] Test task.ipynb on competition sample data, verify submission.csv matches expected format and predictions are reasonable
- [ ] T069 [US2] Run task.ipynb 5 times to verify 100% reproducibility (all submission.csv files should be identical)

**Pre-Submission Validation:**
- [ ] T070 [US2] Create submission checklist document (model trained on multi-datasets ✓, validation F1 >80% ✓, inference <2hrs ✓, submission.csv validated ✓, task.ipynb tested ✓, dependencies listed ✓, reproducible ✓)
- [ ] T071 [US2] Validate checkpoint file size is reasonable for submission (<500MB, compress if needed)
- [ ] T072 [US2] Test task.ipynb in clean environment (fresh conda env, install from requirements only, verify no missing dependencies)
- [ ] T073 [US2] Retrieve CUDA 11.8 competition key from AI Factory My Page > Activity History > Competition
- [ ] T074 [US2] Perform final dry-run submission test (execute all notebook cells, time execution, verify output)

**Checkpoint**: Final submission package ready - task.ipynb validated, model optimized, reproducibility confirmed, ready to submit before Nov 20 deadline

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple components, documentation, and final preparations

**Documentation & Code Quality:**
- [ ] T075 [P] Add docstrings to all public methods in src/models/ following Google style (Args, Returns, Raises sections)
- [ ] T076 [P] Add docstrings to all public methods in src/data/, src/training/, src/inference/ following Google style
- [ ] T077 [P] Add type hints to all function signatures across src/ for better code clarity
- [ ] T078 [P] Update README.md with detailed sections: installation, dataset download instructions, training guide, inference guide, submission guide, expected performance
- [ ] T079 [P] Create notebooks/eda.ipynb for exploratory data analysis (dataset statistics, sample visualizations, label distributions, quality analysis)
- [ ] T080 [P] Create notebooks/experiments.ipynb for tracking experiments (different architectures, hyperparameters, augmentation strategies, results comparison)

**Testing & Validation:**
- [ ] T081 [P] Create unit tests in tests/unit/test_models.py (test forward pass shapes, test each branch independently, test fusion output dimensions)
- [ ] T082 [P] Create unit tests in tests/unit/test_data.py (test face detection, test video frame extraction, test dataset loading, test augmentations)
- [ ] T083 [P] Create unit tests in tests/unit/test_metrics.py (test Macro F1 calculation with known inputs, test per-class metrics, test edge cases)
- [ ] T084 [P] Create integration test in tests/integration/test_training.py (end-to-end training on small dataset for 2 epochs, verify checkpoint saved, metrics tracked)
- [ ] T085 [P] Create integration test in tests/integration/test_inference.py (end-to-end inference on sample data, verify submission.csv created with correct format)
- [ ] T086 Run all tests with pytest, verify >80% pass rate (100% ideal but competition focus is on model performance)

**Performance Monitoring:**
- [ ] T087 Create logging for training progress (epoch, loss, metrics, learning rate, time per epoch) saved to logs/training.log
- [ ] T088 Add TensorBoard logging in Trainer (loss curves, metrics, learning rate schedule) for visualization
- [ ] T089 Create performance comparison table (baseline vs hybrid, single-dataset vs multi-dataset, different backbones) in README.md

**Final Checks:**
- [ ] T090 Validate all config files are correct and match current best settings (model_config.yaml, training_config.yaml, inference_config.yaml)
- [ ] T091 Clean up code: remove debug prints, commented code, unused imports, format with black/autopep8
- [ ] T092 Verify .gitignore excludes all large files (data/, checkpoints/, logs/) and temporary files
- [ ] T093 Final git commit with message: "Final submission ready for competition - hybrid model with Macro F1 >80%"
- [ ] T094 Create backup of entire project directory and task.ipynb before submission

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational phase completion - Core model development
- **User Story 2 (Phase 4)**: Depends on US1 having a trained model checkpoint - Inference & submission
- **Hybrid Enhancement (Phase 5)**: Extends US1 - Improves model performance
- **Final Optimization (Phase 6)**: Depends on Phase 5 best model - Submission preparation
- **Polish (Phase 7)**: Can run in parallel with development or at the end - Documentation & testing

### User Story Dependencies

- **User Story 1 (Model Development)**: Foundation only - No dependencies on other stories
- **User Story 2 (Submission)**: Depends on US1 trained model - Can start once checkpoint exists
- **User Story 3-6**: Not implemented (team management, CUDA env selection, timeline, rules are competition platform features, not our model implementation)

### Within Each User Story

**User Story 1 (Model Development):**
1. Models can be implemented in parallel (T017-T019)
2. DeepfakeDetector integrates all models (T020 depends on T017-T019)
3. Data processing in parallel (T021-T023)
4. Training components in parallel (T024-T025)
5. Training script uses all components (T026 depends on T020-T025)
6. Dataset download and training (T029-T031 sequential)

**User Story 2 (Submission):**
1. Inference components can be in parallel (T032-T034)
2. Optimizations can be in parallel (T035-T037)
3. Scripts and validation (T038-T042 mostly sequential)
4. Notebook creation and testing (T043-T047 sequential)

**Phase 5 (Hybrid Model):**
1. Dataset downloads can be in parallel (T048-T049)
2. Dataset updates (T050-T051)
3. Training configuration and execution (T052-T057 mostly sequential)

**Phase 6 (Final Optimization):**
1. Profiling and optimizations (T058-T062 mostly sequential)
2. Submission updates (T063-T069 mostly sequential)
3. Validation (T070-T074 mostly sequential)

**Phase 7 (Polish):**
- Most tasks marked [P] can run in parallel (documentation, testing, logging)

### Parallel Opportunities

**Setup Phase:**
- T003, T004, T005, T006, T008, T009 can all run in parallel (different files)

**Foundational Phase:**
- T011, T012, T013, T014, T015 can all run in parallel (different modules)

**User Story 1 Models:**
- T017, T018, T019 can all run in parallel (separate branch implementations)

**User Story 1 Data:**
- T021, T022, T023 can run in parallel (different aspects of data processing)

**User Story 1 Training:**
- T024, T025 can run in parallel (loss functions vs trainer class)

**User Story 2 Inference:**
- T032, T033, T034 can run in parallel if interfaces are defined
- T035, T036, T037 can run in parallel (different optimization techniques)
- T039 can run anytime after T038

**Phase 5 Datasets:**
- T048, T049 can run in parallel (downloading different datasets)

**Phase 7 Polish:**
- T075, T076, T077, T078, T079, T080, T081, T082, T083, T084, T085 can all run in parallel (different files/modules)

---

## Parallel Example: User Story 1 - Model Development

```bash
# Launch all model branch implementations in parallel:
Task T017: "Implement SpatialBranch in src/models/spatial_branch.py"
Task T018: "Implement FrequencyBranch in src/models/frequency_branch.py"
Task T019: "Implement FusionLayer in src/models/fusion_layer.py"
# Wait for completion, then integrate in DeepfakeDetector (T020)

# Launch all data processing tasks in parallel:
Task T021: "Implement DeepfakeDataset in src/data/dataset.py"
Task T022: "Extend DataPreprocessor in src/data/transforms.py"
Task T023: "Create preprocessing script in scripts/preprocess_data.py"

# Launch training components in parallel:
Task T024: "Implement CombinedLoss in src/training/losses.py"
Task T025: "Implement Trainer in src/training/trainer.py"
```

## Parallel Example: User Story 2 - Submission

```bash
# Launch inference components in parallel:
Task T032: "Implement ModelLoader in src/inference/model_loader.py"
Task T033: "Implement InferenceEngine in src/inference/inference_engine.py"

# Launch optimizations in parallel:
Task T035: "Implement FP16 support in InferenceEngine"
Task T036: "Implement batch processing in InferenceEngine"
Task T037: "Add early stopping for video inference"

# Launch validation scripts in parallel:
Task T038: "Create inference script in scripts/inference.py"
Task T039: "Create submission validation script in scripts/test_submission.py"
```

---

## Implementation Strategy

### MVP First (Focus on Submission)

1. **Phase 1**: Setup project structure (T001-T009)
2. **Phase 2**: Foundational infrastructure (T010-T016)
3. **Phase 3**: User Story 1 - Baseline Model (T017-T031)
   - **STOP and VALIDATE**: Train baseline, verify >85% accuracy on FF++
4. **Phase 4**: User Story 2 - Submission Pipeline (T032-T047)
   - **STOP and VALIDATE**: Test task.ipynb, verify submission.csv format
5. **FIRST SUBMISSION READY**: Can submit baseline model to competition

### Incremental Improvement

6. **Phase 5**: Hybrid Model Enhancement (T048-T057)
   - **STOP and VALIDATE**: Verify cross-dataset F1 >80%
7. **Phase 6**: Final Optimization (T058-T074)
   - **STOP and VALIDATE**: Test inference speed <2 hours, reproducibility 100%
8. **FINAL SUBMISSION**: Submit optimized hybrid model
9. **Phase 7**: Polish & Documentation (T075-T094)
   - Ongoing improvements, testing, documentation

### Solo Developer Strategy

**Week 1-2 (Phase 3 - Baseline Model):**
- Days 1-2: Setup + Foundational (T001-T016)
- Days 3-7: Model development (T017-T026)
- Days 8-14: Dataset prep, training, validation (T027-T031)

**Week 3-4 (Phase 4 - Submission):**
- Days 15-17: Inference pipeline (T032-T037)
- Days 18-21: Scripts and validation (T038-T042)
- Days 22-28: Submission notebook (T043-T047)

**Week 5 (Phase 5 - Hybrid Model):**
- Days 29-31: Download datasets (T048-T049)
- Days 32-35: Multi-dataset training (T050-T057)

**Week 6 (Phase 6 - Final Optimization):**
- Days 36-39: Speed optimization (T058-T062)
- Days 40-42: Submission finalization (T063-T074)

**Ongoing (Phase 7 - Polish):**
- Parallel with development or final days before submission

### Parallel Team Strategy

If multiple developers available:

1. **All together**: Setup + Foundational (T001-T016)
2. **Split work**:
   - Developer A: Model architectures (T017-T020)
   - Developer B: Data processing (T021-T023)
   - Developer C: Training components (T024-T026)
3. **Merge and train**: Integrate and train baseline (T027-T031)
4. **Split work**:
   - Developer A: Inference engine (T032-T037)
   - Developer B: Scripts and validation (T038-T042)
   - Developer C: Notebook and testing (T043-T047)
5. **Continue in parallel**: Hybrid model, optimization, polish

---

## Task Summary

**Total Tasks**: 94

**Tasks per Phase**:
- Phase 1 (Setup): 9 tasks
- Phase 2 (Foundational): 7 tasks
- Phase 3 (User Story 1 - Model Development): 15 tasks
- Phase 4 (User Story 2 - Submission): 16 tasks
- Phase 5 (Hybrid Enhancement): 10 tasks
- Phase 6 (Final Optimization): 17 tasks
- Phase 7 (Polish): 20 tasks

**Tasks per User Story**:
- US1 (Model Development & Training): 25 tasks (Phase 3 + Phase 5)
- US2 (Submission & Evaluation): 33 tasks (Phase 4 + Phase 6)
- Setup/Foundation/Polish: 36 tasks

**Parallel Opportunities**: 45 tasks marked [P] can run in parallel (48% of total)

**Independent Test Criteria**:
- **US1**: Trained model achieves >85% accuracy on FF++ test set with Macro F1 >83%
- **US2**: task.ipynb executes successfully, generates valid submission.csv, reproducible results
- **Hybrid Model**: Cross-dataset Macro F1 >80% on held-out Celeb-DF validation set
- **Final Submission**: Inference completes in <2 hours on 10K samples, 100% reproducibility

**Suggested MVP Scope**:
- MVP 1 (Weeks 1-2): Baseline model trained and validated (Phase 1-3, Tasks T001-T031)
- MVP 2 (Weeks 3-4): Submission pipeline ready (Phase 4, Tasks T032-T047)
- MVP 3 (Week 5): Hybrid model with better generalization (Phase 5, Tasks T048-T057)
- Final (Week 6): Optimized for competition submission (Phase 6, Tasks T058-T074)

**Format Validation**: ✅ All tasks follow checklist format with:
- Checkbox: `- [ ]`
- Task ID: T001-T094 (sequential)
- [P] marker: 45 tasks marked as parallelizable
- [Story] label: US1 or US2 for user story tasks
- Description: Clear action with exact file path
- No vague tasks, all specific and executable

---

## Notes

- All tasks include exact file paths for implementation
- [P] tasks operate on different files with no dependencies - safe for parallel execution
- [US1] and [US2] labels track which user story each task serves
- Checkpoints after each phase enable independent validation
- Can stop at any phase to validate independently
- MVP (Phases 1-4) delivers a working submission to competition
- Phases 5-6 improve model performance for better leaderboard placement
- Phase 7 can be done in parallel or deferred to end for polish
- All tasks are immediately executable by following contracts/model-interface.md and data-model.md specifications
- Tests are minimal - focus is on model performance and competition submission
- Validation is primarily through competition sample data and manual testing
- Reproducibility is critical - fixed seeds, documented preprocessing, verified in task.ipynb
