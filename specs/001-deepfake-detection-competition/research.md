# Research: State-of-the-Art Deepfake Detection Techniques

**Feature**: Deepfake Detection AI Competition Platform
**Branch**: `001-deepfake-detection-competition`
**Created**: 2025-11-17
**Research Period**: January 2023 - November 2024

## Executive Summary

This research document synthesizes state-of-the-art deepfake detection techniques from recent academic literature (2023-2024) to inform the implementation of a competitive deepfake detection model for the National Forensic Service AI Competition. The competition requires:

- Binary classification (Real=0, Fake=1) for face-based images (JPG/PNG) and videos (MP4)
- Macro F1-score as primary evaluation metric
- Single model architecture (no ensembles)
- 3-hour inference time limit
- Support for diverse deepfake generation methods (face swap, lip sync, commercial/open-source models)

## 1. Current State-of-the-Art Performance (2024)

### 1.1 Benchmark Performance

**Deepfake-Eval-2024 Benchmark** (In-the-wild deepfakes from social media):
- **Leading Commercial Detector**: F1 = 0.83, Accuracy = 82%, AUC = 0.90, Recall = 71%, Precision = 0.99
- **Compass Vision (Top Performer)**: Accuracy = 86.7%
- **Academic Models**: Significant performance drop on real-world data (AUC decreased by 45-50% vs. academic benchmarks)

**Academic Datasets**:
- **FaceForensics++ & DFDC**: F1 = 88.0%
- **DFDC Dataset**: F1 = 91.9% (reported by top methods)
- **Celeb-DF v2**: F1 = 92.52% (using ViViT architecture with facial landmarks)
- **DFT-based methods**: F1 > 99% (on specific datasets, may not generalize)

### 1.2 Key Insight

There is a significant generalization gap between performance on academic datasets versus real-world deepfakes. Cross-dataset robustness is critical for competition success.

## 2. Recommended Architecture Components

### 2.1 Backbone Networks

**Primary Recommendation: EfficientNet Family**

**Rationale**:
- All DFDC Challenge winners used pre-trained EfficientNet networks
- EfficientNet-B7 ensemble techniques achieved top results on DFDC
- Excellent balance of accuracy and computational efficiency
- Strong transfer learning capabilities

**Implementation Variants**:
1. **EfficientNet-B4**: Good balance of speed and accuracy (recommended for 3-hour constraint)
2. **EfficientNet-B7**: Highest accuracy but slower (consider for image-only branch)
3. **EfficientNet-V2**: Latest version with improved training speed

**Alternative Backbones** (for comparison during development):
- **Xception**: Strong performance on facial manipulation detection
- **MobileNet**: Faster inference for video processing
- **Swin Transformer**: Strong spatial-temporal modeling

### 2.2 Hybrid Architectures (Highly Recommended)

Recent research (2024) shows hybrid models outperform pure CNN or pure Transformer approaches:

**Option 1: EfficientNet + Vision Transformer (Cross-ViT)**
- Use EfficientNet for local feature extraction
- Apply Vision Transformer encoder for global context modeling
- Achieved 92.4% AUC on FF++, 83.1% on Celeb-DF v2

**Option 2: EfficientNet + Frequency Domain Learning (FreqNet-style)**
- Spatial branch: EfficientNet backbone
- Frequency branch: FFT/DCT-based high-frequency artifact detection
- Fusion layer combining spatial and frequency features
- State-of-the-art generalization (+9.8% improvement)

**Option 3: CNN-LSTM-Transformer Hybrid (for videos)**
- EfficientNet/ResNet for frame-level feature extraction
- LSTM for temporal consistency modeling
- Transformer for global attention across frames
- Fast inference speed suitable for time-sensitive scenarios

### 2.3 Frequency Domain Analysis (Critical for Generalization)

**Why Frequency Domain?**
- Deepfake artifacts stem from GAN upsampling operations
- High-frequency spectral anomalies persist across compression levels
- Better generalization to unseen manipulation methods

**Recommended Frequency Techniques**:

1. **FreqNet Approach** (AAAI 2024):
   - Apply FFT to convert images to frequency domain
   - Process amplitude spectrum and phase spectrum separately
   - Convolutional layers in frequency space before IFFT
   - Achieved +9.8% generalization improvement with fewer parameters
   - **Implementation Complexity**: Medium - requires FFT/IFFT operations, dual-channel processing
   - **Inference Overhead**: ~15-20ms additional per image (on top of baseline EfficientNet)

2. **F3-Net (DCT-based)**:
   - Separate frequency components via Discrete Cosine Transform
   - Extract local frequency statistics
   - Robust on compressed videos (critical for real-world data)
   - **Implementation Complexity**: Medium-High - DCT computation, local frequency feature extraction
   - **Inference Overhead**: ~20-25ms additional per image
   - **Trade-off**: Better robustness to JPEG compression vs higher computational cost

3. **HiFE (High-Frequency Enhancement)**:
   - Adaptive local and global high-frequency enhancement branches
   - No need for uncompressed content supervision
   - Handles highly compressed content well
   - **Implementation Complexity**: High - requires adaptive filtering, dual-branch high-frequency processing
   - **Inference Overhead**: ~25-30ms additional per image
   - **Trade-off**: Best compression robustness but slower; may not be necessary if supervised labels available

**Implementation Strategy for Competition**:

**Recommended Approach - Hybrid Architecture Integration**:
- **Dual-branch architecture**: Spatial RGB (EfficientNet-B4) + Frequency (FFT or DCT)
- **Fusion Method**: Choose between:
  1. **Late Fusion** (Simpler): Concatenate spatial + frequency features → FC layers → Classifier
     - Pros: Easier to implement, faster convergence
     - Cons: Limited cross-modal interaction
  2. **Hierarchical Cross-Modal Fusion** (Better Performance): Multi-level feature fusion with attention
     - Pros: Better feature interaction, +2-3% F1 improvement
     - Cons: More complex, requires careful tuning
- **Processing**: Can parallelize spatial and frequency branches on GPU for speed
- **Expected Total Inference Time**: 40-60ms per image (EfficientNet-B4: 20-30ms + Frequency: 15-25ms + Fusion: 5-10ms)

**Selection Criteria for Frequency Technique**:
- **If training on high-quality data (FF++, DFDC)**: FreqNet (FFT-based) - best generalization, reasonable speed
- **If test data expected to be compressed**: F3-Net (DCT-based) - handles JPEG artifacts better
- **If computational budget is tight**: Skip frequency branch or use lightweight FFT-only approach

**Performance vs Complexity Trade-offs**:
| Method | Generalization Gain | Inference Time | Implementation Effort | Recommended for Competition |
|--------|--------------------:|---------------:|----------------------:|----------------------------:|
| FreqNet (FFT) | +9.8% | +15-20ms | Medium | ✓ Yes (Primary choice) |
| F3-Net (DCT) | +7-8% | +20-25ms | Medium-High | Maybe (if compression is major concern) |
| HiFE | +10-12% | +25-30ms | High | No (too slow for 3-hour constraint) |
| Late Fusion | Baseline | +5-10ms | Low | ✓ Yes (faster alternative) |
| Hierarchical Fusion | +2-3% over late | +10-15ms | Medium | ✓ Yes (if time permits) |

## 3. Data Preprocessing & Augmentation

### 3.1 Face Detection and Alignment

**Recommended Pipeline** (DeepfakeBench standard):

1. **Face Detection**:
   - Use RetinaFace or MTCNN (robust to various poses)
   - Sequential "waterfall" architecture: multi-level modular detectors in increasing complexity
   - Extract bounding box with margin (include context around face)

2. **Face Alignment**:
   - Detect 68 facial landmarks (dlib or similar)
   - Apply affine transformation to standardize pose
   - Optional but improves consistency

3. **Cropping & Resizing**:
   - Recommended resolution: 224×224 (EfficientNet standard) or 384×384 (higher detail)
   - Use "Face method": extract face with margin to include manipulation artifacts near face boundary

4. **Save Preprocessing Artifacts**:
   - Store landmarks for potential use in model
   - Document all preprocessing steps for reproducibility (required for verification)

### 3.2 Data Augmentation (Essential for Generalization)

**Standard Augmentations** (apply during training):

1. **Geometric Transformations**:
   - Horizontal flip (p=0.5)
   - Random rotation (±15°)
   - Random affine transformations
   - Random crop and resize
   - Shear transformation

2. **Color/Appearance**:
   - Random brightness/contrast (p=0.5)
   - Random hue/saturation adjustment
   - Gamma adjustment
   - HSV-based color jitter
   - Grayscale conversion (p=0.2)
   - Fancy PCA color augmentation (p=0.5)

3. **Compression & Noise** (Critical for Robustness):
   - JPEG compression with quality 60-100 (p=0.5)
   - Gaussian noise (p=0.1, σ=0.01-0.05)
   - Gaussian blur (p=0.05, kernel 3-7)
   - Isotropic resizing (downscale + upscale, p=1.0)

4. **Advanced Augmentation** (2024 Best Practices):
   - **RandAugment** with rand-m9-mstd0.5-inc1 configuration
   - 15 operations: contrast adjustment, histogram equalization, rotation, shear, etc.
   - Automatically searches for optimal augmentation policy

**Augmentation Strategy**:
- Use stronger augmentation for fake samples (data imbalance correction)
- Apply compression-based augmentation to match real-world degradation
- Validate augmentation impact on Macro F1-score (not just accuracy)

### 3.3 Video Frame Sampling

**Challenge**: 5-second videos at 30fps = ~150 frames. Processing all frames is computationally expensive.

**Recommended Sampling Strategies** (2024 Research):

1. **Dynamic P-Value Sampling** (Efficiency):
   - Process frames sequentially
   - Compute confidence using statistical t-test
   - Stop early when p-value threshold is reached
   - Reduces frames processed while maintaining accuracy

2. **Snippet-based Sampling** (Temporal Consistency):
   - Extract "snippets" of consecutive frames (e.g., 8-16 frames)
   - Learn local temporal inconsistencies between neighboring frames
   - Better than uniform sampling for detecting video artifacts

3. **Adaptive Content-Aware Sampling**:
   - Compare frames pixel-wise to identify significant changes
   - Sample frames with high motion or scene changes
   - Skip redundant frames (static scenes)

4. **Multi-Scale Temporal Sampling**:
   - Sample 8 frames from first 32 frames (early detection)
   - Sample 8 frames evenly across entire video (global view)
   - Sample 8 frames from first 300 frames (balanced)
   - Ensemble predictions from different sampling strategies

**Recommended for Competition**:
- **Primary**: Sample 16-32 frames uniformly across 5-second video
- **Fallback**: Dynamic sampling if inference time is critical
- **Per-Frame Prediction**: Average or max-pooling across frames for video-level label

## 4. Training Strategies for High Macro F1-Score

### 4.1 Loss Functions

**Standard Cross-Entropy** is suboptimal for Macro F1-score. Consider:

1. **Focal Loss** (address class imbalance if present):
   ```
   FL(pt) = -α(1-pt)^γ log(pt)
   ```
   - γ=2, α=0.25 (standard settings)
   - Focuses on hard examples

2. **Macro F1 Loss** (directly optimize target metric):
   - Differentiable approximation of F1-score
   - Calculate F1 for each class separately, then average

3. **Combined Loss**:
   ```
   L_total = λ1 * L_CE + λ2 * L_focal + λ3 * L_F1
   ```
   - Start with CE, gradually increase F1 loss weight

### 4.2 Class Balancing

- Use **weighted sampling** to ensure equal representation of Real/Fake
- Monitor per-class metrics (Precision, Recall for both classes)
- Optimize for balanced Macro F1 (not just accuracy)

### 4.3 Training Schedule

1. **Warmup Phase** (5-10 epochs):
   - Low learning rate (1e-5)
   - Freeze backbone, train only classifier head
   - Use standard CE loss

2. **Main Training** (50-100 epochs):
   - Unfreeze all layers
   - Learning rate: 1e-4 to 5e-5 (with cosine annealing)
   - Apply full augmentation
   - Use combined loss function

3. **Fine-tuning Phase** (10-20 epochs):
   - Lower learning rate (1e-5 to 1e-6)
   - Reduce augmentation strength
   - Focus on F1 optimization

### 4.4 Multi-Dataset Training (Generalization)

**Recommended Approach**:
- Train on multiple public datasets: FaceForensics++, DFDC, Celeb-DF
- Use domain adaptation techniques or multi-domain learning
- Validate on held-out dataset to measure generalization

**Why Important**:
- Competition test data may differ from any single training dataset
- Cross-dataset robustness strongly correlates with real-world performance

## 5. Generalization & Robustness Techniques

### 5.1 Critical Techniques from 2024 Research

**1. Consistency Regularization (CORE Method)**:
- Enforce consistency in learned features across transformations
- Add consistency loss: minimize feature distance between original and augmented samples
- Model focuses on manipulation-relevant features (not dataset-specific artifacts)

**2. Multi-Scale Attention Mechanisms**:
- Detect forged areas at multiple scales
- Reduces reliance on global identity features
- Improves generalization to subtle alterations

**3. Frequency-Spatial Fusion**:
- Combine spatial RGB features with frequency domain features
- Frequency features are more robust to compression and post-processing
- Use learned fusion weights or hierarchical cross-modal fusion

**4. Latent Space Augmentation** (2024 CVPR):
- Augment in feature space (not just input space)
- Transcends forgery-specific patterns
- Improves robustness to unseen manipulation types

**5. Disentangled Representation Learning**:
- Separate demographic features from forgery features
- Ensures fairness across different face attributes
- Improves generalization across diverse test data

### 5.2 Adversarial Robustness

**Not strictly required for competition**, but useful for edge cases:

- Adversarial training with PGD/FGSM attacks (mild perturbations)
- Gradient masking prevention
- Improves robustness to subtle manipulations

## 6. Inference Optimization (3-Hour Constraint)

### 6.1 Speed Optimization Techniques

**Model-Level**:
1. Use **EfficientNet-B4** instead of B7 (2x faster, minimal accuracy loss)
2. Apply **knowledge distillation**: Train large model, distill to smaller student model
3. Use **mixed precision inference** (FP16) for 2x speedup
4. **Batch processing**: Process multiple images/frames in parallel

**Video-Specific**:
1. **Frame sampling**: Process 16-32 frames instead of all frames
2. **Early stopping**: Use confidence thresholding to skip remaining frames
3. **Parallel processing**: Process video frames in parallel (GPU batch)

**System-Level**:
1. Pre-load model weights (avoid repeated loading)
2. Use efficient data loading (PyTorch DataLoader with num_workers=8)
3. Minimize I/O operations (process in-memory when possible)
4. Use GPU acceleration (CUDA optimization)

### 6.2 Estimated Inference Times

Assuming L4 GPU (or T4):

- **EfficientNet-B4**: ~20-30ms per image (224×224)
- **EfficientNet-B7**: ~50-70ms per image
- **Hybrid (EfficientNet + Transformer)**: ~40-60ms per image
- **Video (16 frames)**: ~0.5-1 second per video

For ~10,000 test samples (mixed images/videos):
- **Conservative estimate**: 1-2 hours (well within 3-hour limit)
- **Headroom**: Allows for larger models or more frames

## 7. Recommended Pipeline Architecture

### 7.1 Overall Architecture

```
Input (Image/Video) → Preprocessing → Dual-Branch Model → Fusion → Binary Classification
```

**Dual-Branch Model**:

**Branch 1: Spatial Features**
- Backbone: EfficientNet-B4 (pretrained on ImageNet)
- Additional layers: Vision Transformer encoder (4 layers)
- Output: 512-dim spatial feature vector

**Branch 2: Frequency Features**
- FFT transformation (or DCT)
- Process amplitude and phase spectra separately
- Convolutional layers in frequency domain
- Output: 512-dim frequency feature vector

**Fusion Layer**:
- Concatenate spatial + frequency features (1024-dim)
- Self-attention mechanism (learn importance weights)
- Fully connected layers
- Output: Binary prediction (Real=0, Fake=1)

### 7.2 Separate Models for Images vs. Videos (Allowed by Competition)

**Image Model**:
- Larger backbone (EfficientNet-B7) for maximum accuracy
- Single frame processing
- Higher resolution (384×384)

**Video Model**:
- Faster backbone (EfficientNet-B4) for speed
- Frame sampling + temporal aggregation (LSTM or Transformer)
- Lower resolution (224×224) to process more frames
- Output aggregation: Average logits across frames

### 7.3 Training Data Construction

**Critical: Build Diverse Training Dataset**

Competition rules allow using publicly available data. Recommended sources:

1. **FaceForensics++** (FF++):
   - 1000 real videos, 4000 fake videos
   - Methods: Deepfakes, Face2Face, FaceSwap, NeuralTextures
   - High quality + compressed versions

2. **Deepfake Detection Challenge (DFDC)**:
   - 124,000 videos (largest dataset)
   - Diverse actors, scenarios, manipulation methods

3. **Celeb-DF v2**:
   - 590 real + 5639 fake celebrity videos
   - High visual quality, challenging detection

4. **DeeperForensics-1.0**:
   - 60,000 videos with diverse variations
   - Occlusion, lighting, pose changes

5. **Synthetic Data Generation** (optional):
   - Use open-source deepfake tools (e.g., faceswap, SimSwap)
   - Generate fakes from real videos to match test distribution

**Data Mixing Strategy**:
- Train on all datasets with equal sampling
- Validate on held-out portion of each dataset
- Monitor cross-dataset performance

## 8. Implementation Recommendations

### 8.1 Technology Stack

**Framework**: PyTorch 1.8+ (CUDA 11.8 environment recommended)

**Key Libraries**:
- `torch`, `torchvision`: Deep learning framework
- `timm`: EfficientNet and ViT implementations
- `opencv-python-headless`: Video processing, face detection
- `albumentations`: Advanced data augmentation
- `facenet-pytorch` or `retinaface-pytorch`: Face detection
- `numpy`, `scipy`: Numerical operations, FFT/DCT
- `pandas`: CSV handling for submission.csv

**Face Detection**:
- RetinaFace (recommended for robustness)
- MTCNN (lightweight alternative)
- MediaPipe (Google's solution, very fast)

### 8.2 Development Phases

**Phase 1: Baseline Model (1-2 weeks)**
- Implement EfficientNet-B4 baseline
- Train on FaceForensics++ only
- Achieve >90% accuracy on FF++ test set
- Generate submission.csv in correct format

**Phase 2: Hybrid Architecture (1-2 weeks)**
- Add frequency branch (FFT/DCT)
- Implement fusion layer
- Train on multi-dataset (FF++ + DFDC + Celeb-DF)
- Target: 85%+ F1 on cross-dataset validation

**Phase 3: Optimization (1 week)**
- Optimize for Macro F1-score (not just accuracy)
- Fine-tune loss functions and class balancing
- Implement video frame sampling and aggregation
- Ensure inference completes within 1-2 hours

**Phase 4: Robustness & Submission (1 week)**
- Add consistency regularization
- Test on compressed/augmented data
- Verify reproducibility of preprocessing
- Create final task.ipynb with all dependencies

### 8.3 Validation Strategy

**Critical for Competition Success**:

1. **K-Fold Cross-Validation** (5-fold):
   - Ensures model generalizes within training data
   - Reduces overfitting risk

2. **Cross-Dataset Validation**:
   - Train on FF++ + DFDC, validate on Celeb-DF
   - Rotate datasets to measure generalization

3. **Macro F1 Tracking**:
   - Monitor Macro F1 (not accuracy) on validation set
   - Track per-class Precision/Recall
   - Ensure balanced performance on Real and Fake classes

4. **Compression Testing**:
   - Test on JPEG-compressed validation data (quality 70-90)
   - Simulates real-world degradation

## 9. Risk Mitigation

### 9.1 Known Challenges

**Challenge 1: Generalization Gap**
- **Risk**: Model overfits to training data distribution
- **Mitigation**: Multi-dataset training, strong augmentation, frequency domain features

**Challenge 2: Compression Artifacts**
- **Risk**: Test data may be heavily compressed
- **Mitigation**: JPEG augmentation, high-frequency enhancement (HiFE)

**Challenge 3: Class Imbalance**
- **Risk**: Unequal Real/Fake distribution in test data
- **Mitigation**: Optimize for Macro F1 (not accuracy), use balanced sampling

**Challenge 4: Inference Time Limit**
- **Risk**: Model too slow for 3-hour limit
- **Mitigation**: EfficientNet-B4, frame sampling, batch processing, early testing

**Challenge 5: Reproducibility Verification**
- **Risk**: Score cannot be reproduced during verification period
- **Mitigation**: Document all preprocessing, fix random seeds, test locally multiple times

### 9.2 Fallback Strategies

**If Hybrid Model Too Slow**:
- Fall back to pure EfficientNet-B7 (still competitive)
- Reduce frame count for videos
- Use lower input resolution (224×224 instead of 384×384)

**If Generalization Poor**:
- Increase augmentation strength
- Add more datasets to training mix
- Simplify model (reduce overfitting)

**If Macro F1 Imbalanced**:
- Adjust loss function weights
- Use threshold optimization (find optimal decision boundary)

## 10. Expected Performance

### 10.1 Realistic Performance Targets

Based on 2024 research and competition constraints:

**Baseline (EfficientNet-B4, single dataset)**:
- Internal validation F1: 88-92%
- Cross-dataset F1: 75-80%

**Hybrid Model (EfficientNet + Frequency, multi-dataset)**:
- Internal validation F1: 90-94%
- Cross-dataset F1: 82-87%
- **Competition test F1 (estimated)**: 80-85%

**Optimistic (with all optimizations)**:
- Competition test F1: 85-90%

### 10.2 Success Criteria

**Minimum Viable Model**:
- Macro F1 > 75% on competition test set
- Inference completes within 3 hours
- Reproducible results during verification

**Competitive Model**:
- Macro F1 > 82% (likely top 10%)
- Balanced precision/recall on both classes
- Robust to compression and unseen manipulation methods

**Award-Winning Model**:
- Macro F1 > 88% (likely top 3)
- Superior generalization across all test samples
- Novel techniques or exceptional implementation quality

## 11. References & Resources

### 11.1 Key Papers (2023-2024)

1. **FreqNet** (AAAI 2024): Frequency-Aware Deepfake Detection
   - Introduces frequency domain learning for generalization

2. **Deepfake-Eval-2024**: Multi-Modal In-the-Wild Benchmark
   - Real-world performance evaluation, identifies generalization gap

3. **HiFE** (2024): High-Frequency Enhancement Network
   - Handles compressed deepfakes without supervision

4. **CORE** (2024): Consistency Regularization for Deepfake Detection
   - Cross-dataset robustness through consistency learning

5. **GM-DF** (2024): Generalized Multi-Scenario Deepfake Detection
   - Hybrid expert modeling for domain-specific features

### 11.2 Code Resources

1. **DeepfakeBench**: Comprehensive benchmark toolkit
   - GitHub: SCLBD/DeepfakeBench
   - Standardized preprocessing, evaluation

2. **timm Library**: PyTorch Image Models
   - Pre-trained EfficientNet, ViT models
   - Easy fine-tuning interface

3. **Albumentations**: Advanced augmentation library
   - Efficient implementation of all recommended augmentations

### 11.3 Datasets

1. FaceForensics++: https://github.com/ondyari/FaceForensics
2. DFDC: https://ai.facebook.com/datasets/dfdc/
3. Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics
4. DeeperForensics: https://github.com/EndlessSora/DeeperForensics-1.0

## 12. Conclusion

The recommended approach combines:

1. **Hybrid Architecture**: EfficientNet backbone + Frequency domain analysis
2. **Multi-Dataset Training**: FF++ + DFDC + Celeb-DF for generalization
3. **Strong Augmentation**: Compression, noise, color, geometric transformations
4. **Macro F1 Optimization**: Direct optimization of target metric
5. **Efficient Inference**: Frame sampling, batch processing, optimized backbone

This research-backed strategy balances accuracy, generalization, and computational efficiency to create a competitive deepfake detection model that complies with all competition rules while leveraging state-of-the-art techniques from 2023-2024 literature.

**Next Steps**: Proceed to Phase 1 (Design & Contracts) to translate this research into concrete implementation specifications.
