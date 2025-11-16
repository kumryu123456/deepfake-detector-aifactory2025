# Feature Specification: Deepfake Detection AI Competition Platform

**Feature Branch**: `001-deepfake-detection-competition`
**Created**: 2025-11-17
**Status**: Draft
**Input**: User description: "딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회 - 얼굴 이미지 및 동영상에서 실제와 가짜를 판별하는 이진 분류 모델 개발. 비공개 테스트셋을 활용한 추론 자동화 시스템으로 평가하며, Macro F1-score를 주요 지표로 사용합니다."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Model Development and Training (Priority: P1)

A participant downloads sample data, trains a deepfake detection model using their own dataset, and prepares their model for submission. They need to develop a binary classifier that can detect fake faces in both images (JPG/PNG) and videos (MP4).

**Why this priority**: This is the core activity of the competition - without the ability to develop and train models, participants cannot compete. This represents the primary value proposition.

**Independent Test**: Can be fully tested by providing sample data download, verifying participants can access training resources, and confirming they can develop a working model locally before submission.

**Acceptance Scenarios**:

1. **Given** a participant has registered for the competition, **When** they access the data tab, **Then** they can download sample fake images (7 samples) and fake videos (5 samples)
2. **Given** a participant needs training data, **When** they review the competition guidelines, **Then** they understand they must build their own training dataset using publicly available data
3. **Given** a participant has trained a model, **When** they test it locally on sample data, **Then** their model outputs binary predictions (0 for Real, 1 for Fake)
4. **Given** a participant processes a video file, **When** they extract frames, **Then** their model produces a single prediction for the entire video
5. **Given** a participant's model processes test data, **When** inference completes, **Then** output includes filename (with extension) and integer label in CSV format

---

### User Story 2 - Model Submission and Automated Evaluation (Priority: P1)

A participant packages their trained model into a task.ipynb notebook with all dependencies, submits it through the aifactory library, and receives automated evaluation results based on Macro F1-score using a private test dataset.

**Why this priority**: Submission and evaluation are equally critical to model development - without these, the competition cannot function. This enables fair, automated scoring.

**Independent Test**: Can be tested by submitting a baseline model through the automated inference system and verifying score calculation, regardless of whether other features are complete.

**Acceptance Scenarios**:

1. **Given** a participant has a trained model, **When** they create a task.ipynb file with model loading and inference code, **Then** the notebook includes library installations, model inference, and result saving
2. **Given** a participant has prepared their submission, **When** they call aifactory.score.submit() with their competition key, **Then** their notebook and all subdirectories are compressed and uploaded
3. **Given** a submission is uploaded, **When** the automated system processes it, **Then** inference runs on private test data in a secure cloud environment (CPU: 8 core, RAM: 48GB, L4 GPU)
4. **Given** inference is running, **When** the model processes test data from ./data/ directory, **Then** results are saved to submission.csv in the current directory
5. **Given** inference completes within 3 hours, **When** results are validated, **Then** Macro F1-score is calculated comparing submission.csv against answer.csv
6. **Given** evaluation completes, **When** score is calculated, **Then** results appear on the leaderboard with timestamp
7. **Given** an error occurs during inference, **When** the participant checks submission history, **Then** they can view the error traceback message

---

### User Story 3 - Team Formation and Collaboration (Priority: P2)

Participants form teams of up to 5 members, collaborate on model development, and submit as a unified team with shared submission quotas and merged results on the leaderboard.

**Why this priority**: Team collaboration enhances competition value but is not essential for individual participation. Teams can improve model quality through diverse expertise.

**Independent Test**: Can be tested by creating a team, inviting members, and verifying submission quotas are shared, independent of scoring or model development features.

**Acceptance Scenarios**:

1. **Given** a registered participant wants to form a team, **When** they access the team management page, **Then** they can create a team and invite up to 4 other members
2. **Given** a team is formed, **When** any member submits a model, **Then** the submission counts toward the team's daily quota (3 submissions per day)
3. **Given** a team member wants to leave before November 6, **When** the team leader sends an exit request and the member approves, **Then** the member's submission history is removed from team records
4. **Given** the team merger deadline (November 6) has passed, **When** a participant tries to join or leave a team, **Then** the system only allows individual submissions
5. **Given** a team has met award requirements, **When** prizes are distributed, **Then** the team leader receives the prize on behalf of the team

---

### User Story 4 - CUDA Environment Selection (Priority: P2)

Participants select from three CUDA environments (10.2, 11.8, 12.6) based on their model's dependencies, ensuring compatibility between local development and submission inference.

**Why this priority**: Environment compatibility is important for successful submissions but secondary to core development and submission workflows. Most participants will use default settings.

**Independent Test**: Can be tested by submitting the same model to different CUDA environments and verifying inference runs successfully in the selected environment.

**Acceptance Scenarios**:

1. **Given** a participant reviews environment options, **When** they check available CUDA versions, **Then** they see three options: CUDA 10.2 (Python 3.8, torch 1.6.0), CUDA 11.8 (Python 3.9, torch 1.8.0), and CUDA 12.6 (Python 3.10, torch 2.7.1)
2. **Given** a participant selects a CUDA environment, **When** they retrieve their competition key, **Then** each CUDA version has a separate competition key
3. **Given** a participant submits with a specific CUDA key, **When** inference starts, **Then** the environment matches the selected CUDA version with corresponding default libraries
4. **Given** a participant needs different library versions, **When** they write task.ipynb, **Then** they can install compatible libraries using pip install commands
5. **Given** library installation completes, **When** inference begins, **Then** network access is disabled to ensure fair evaluation

---

### User Story 5 - Competition Timeline and Milestone Tracking (Priority: P3)

Participants track key dates including registration period, team merger deadline, submission deadline, model verification period, and award announcement to plan their competition strategy.

**Why this priority**: Timeline awareness helps participants plan but doesn't affect core functionality. Dates are informational and can be communicated through documentation.

**Independent Test**: Can be tested by displaying competition timeline and sending notifications at key milestones, independent of other features.

**Acceptance Scenarios**:

1. **Given** a participant views the competition overview, **When** they check the schedule, **Then** they see registration runs September 25 - November 20, 5 PM
2. **Given** the competition period begins, **When** participants start submitting, **Then** submissions are accepted from October 23, 10 AM through November 20, 5 PM
3. **Given** participants are forming teams, **When** they check the team merger deadline, **Then** they see it closes on November 6, 5 PM
4. **Given** the submission deadline passes, **When** the verification period begins November 21, **Then** top submissions undergo reproducibility verification until November 26
5. **Given** verification completes, **When** November 27 arrives, **Then** award winners are announced at 5 PM
6. **Given** award winners are announced, **When** December 3 arrives, **Then** the award ceremony takes place at Osong Convention Center

---

### User Story 6 - Competition Rules and Constraints Compliance (Priority: P2)

Participants understand and comply with competition rules including submission limits, single model requirements, Korean citizenship requirement, inference-only evaluation, and intellectual property agreements.

**Why this priority**: Rules compliance is essential for fair competition but is primarily enforced through system constraints and documentation rather than active workflows.

**Independent Test**: Can be tested by attempting to violate rules (e.g., exceeding submission quota, submitting ensemble models) and verifying the system enforces constraints.

**Acceptance Scenarios**:

1. **Given** a team submits models, **When** they attempt a 4th submission in one day, **Then** the system blocks the submission and displays the daily limit message
2. **Given** a participant submits successfully, **When** they try to resubmit, **Then** they must wait 30 minutes after viewing previous results
3. **Given** a participant's inference runs, **When** execution exceeds 3 hours, **Then** the system automatically terminates the process
4. **Given** a participant's submission uses test data for training, **When** validation detects pseudo-labeling or model tuning on test data, **Then** the submission is rejected
5. **Given** a participant attempts to register, **When** they verify citizenship, **Then** only Korean nationals can participate (including overseas residents)
6. **Given** a participant develops their model, **When** they attempt ensemble voting of multiple models, **Then** the submission guidelines clarify only single models are allowed
7. **Given** award winners are selected, **When** intellectual property transfer begins, **Then** winners sign technology transfer agreements granting the organizer implementation rights
8. **Given** a participant uses licensed data or models, **When** they win an award, **Then** they must disclose model names and license types, accepting legal responsibility

---

### Edge Cases

- What happens when a participant's submission.csv is missing required columns (filename, label)?
- How does the system handle videos shorter or longer than 5 seconds during evaluation?
- What happens when a model produces null/None predictions instead of 0 or 1?
- How does the system respond when a participant's submission exceeds storage limits due to large model files?
- What happens when two teams have identical Macro F1-scores on the leaderboard?
- How are disputes resolved if a participant's score cannot be reproduced during verification?
- What happens when a participant submits using the wrong CUDA environment key for their model dependencies?
- How does the system handle face detection when no face is detected in an image/video (even though guidelines state all data contains one face)?
- What happens if a participant's local environment differs significantly from the inference environment, causing unexpected errors?
- How are timeline adjustments communicated if organizers need to modify dates?

## Requirements *(mandatory)*

### Functional Requirements

#### Data and Content Management

- **FR-001**: System MUST provide downloadable sample dataset containing 7 fake images and 5 fake videos
- **FR-002**: System MUST support test data in formats: JPG, PNG for images and MP4 for videos
- **FR-003**: Test data MUST contain exactly one identifiable person's face per file
- **FR-004**: Video test data MUST have average duration of 5 seconds
- **FR-005**: Test data MUST include diverse ethnicities (Asian, White, Black) and age ranges
- **FR-006**: Test data MUST include outputs from commercial generation services, open-source models, and legacy face swap/lip sync techniques
- **FR-007**: Video test data MUST NOT include audio tracks
- **FR-008**: Evaluation data MUST be stored in ./data/ directory with mixed image and video files (no subdirectories)

#### Model Submission and Inference

- **FR-009**: System MUST accept submissions as task.ipynb Jupyter notebook files
- **FR-010**: Participants MUST package all model files and dependencies in subdirectories relative to task.ipynb
- **FR-011**: System MUST support library installation via pip install commands in notebook cells
- **FR-012**: System MUST compress and upload task.ipynb directory and all subdirectories upon submission
- **FR-013**: System MUST provide three CUDA environments: CUDA 10.2 (Python 3.8), CUDA 11.8 (Python 3.9), CUDA 12.6 (Python 3.10)
- **FR-014**: Each CUDA environment MUST include default libraries: torch, torchvision, numpy, scipy, scikit-learn, opencv-python-headless, pandas, Pillow
- **FR-015**: System MUST disable network access after library installation phase
- **FR-016**: System MUST allocate compute resources: 8-core CPU, 48GB RAM, L4 GPU (or T4 if queue is long)
- **FR-017**: System MUST terminate inference automatically after 3 hours maximum execution time
- **FR-018**: System MUST provide several GB of disk storage for inference workspace
- **FR-019**: Participants MUST NOT include .git folders, personal information, or unnecessary files in submission
- **FR-020**: Inference environment MUST be isolated and secure for confidential test data protection

#### Output Format and Validation

- **FR-021**: Model inference MUST generate submission.csv file in current directory
- **FR-022**: submission.csv MUST contain two columns: filename (string) and label (integer)
- **FR-023**: filename column MUST include file extensions (e.g., "image1.jpg", "video1.mp4")
- **FR-024**: label column MUST contain only integer values: 0 (Real) or 1 (Fake)
- **FR-025**: System MUST accept submission.csv with files in any order (not required to match input order)
- **FR-026**: For video files, model MUST output single prediction per video (not per frame)
- **FR-027**: System MUST reject submissions with null, None, or non-binary label values
- **FR-028**: System MUST prohibit pseudo-labeling, test-time training, or model tuning using evaluation data

#### Scoring and Evaluation

- **FR-029**: System MUST calculate Macro F1-score as primary evaluation metric
- **FR-030**: System MUST define Fake (label 1) as positive class and Real (label 0) as negative class
- **FR-031**: System MUST compare submission.csv labels against answer.csv ground truth
- **FR-032**: System MUST display evaluation results on leaderboard with timestamp
- **FR-033**: System MUST provide error traceback viewing for failed submissions
- **FR-034**: Failed or errored submissions MUST NOT count toward daily submission quota
- **FR-035**: System MUST verify score reproducibility during verification period (November 21-26)
- **FR-036**: System MUST invalidate leaderboard scores that cannot be reproduced during verification
- **FR-037**: Tie-breaking procedures for identical scores MUST be documented [NEEDS CLARIFICATION: specific tie-breaking criteria not specified in materials]

#### Model Validation and Generalization (Development Best Practices)

- **FR-095**: Participants SHOULD implement K-fold cross-validation (recommended: 5-fold) within training datasets to ensure model robustness
- **FR-096**: Participants SHOULD perform cross-dataset validation to measure generalization capability (e.g., train on FaceForensics++ and DFDC, validate on Celeb-DF)
- **FR-097**: Participants SHOULD track per-class metrics (Precision and Recall for both Real and Fake classes) during validation to ensure balanced Macro F1-score performance
- **FR-098**: Participants SHOULD implement early stopping based on validation Macro F1-score (not accuracy) with patience of 10-15 epochs
- **FR-099**: Cross-dataset evaluation protocol SHOULD include: (a) Clear train/validation/test split definitions per dataset, (b) Explicit documentation of which datasets are used for training vs validation, (c) Separate tracking of within-dataset and cross-dataset Macro F1 scores
- **FR-100**: Validation data MUST be held out during training and MUST NOT be used for model selection, hyperparameter tuning, or any form of training
- **FR-101**: Participants SHOULD test model performance on JPEG-compressed validation data (quality 70-90) to simulate real-world degradation
- **FR-102**: Final model selection SHOULD prioritize cross-dataset generalization metrics over single-dataset performance to maximize competition test set performance

#### Team and Participant Management

- **FR-038**: System MUST allow individual participation and team formation (up to 5 members maximum)
- **FR-039**: System MUST require Korean nationality for all participants
- **FR-040**: Minor participants MUST submit parental consent forms to cs@aifactory.page
- **FR-041**: Both team leaders and members MUST be able to submit models
- **FR-042**: System MUST enforce daily submission quota of 3 attempts per team (not per individual)
- **FR-043**: System MUST allow team member removal before November 6, 5 PM deadline
- **FR-044**: Team leader MUST initiate exit requests, and members MUST approve to leave
- **FR-045**: System MUST delete departing member's submission history from team records
- **FR-046**: System MUST prohibit team changes after November 6, 5 PM (only individual submissions allowed)
- **FR-047**: System MUST prohibit rejoining competition after leaving a team (no 1-person teams or new teams allowed)
- **FR-048**: System MUST prohibit sharing code or data outside team boundaries
- **FR-049**: Team leaders MUST receive prizes on behalf of their teams when award criteria are met

#### Submission Quota and Timing

- **FR-050**: System MUST limit submissions to 3 per team per day
- **FR-051**: System MUST enforce 30-minute cooldown between successful submissions (after viewing results)
- **FR-052**: Cooldown timer MUST start only after participant views previous submission results
- **FR-053**: System MUST count only successfully evaluated submissions toward daily quota
- **FR-054**: System MUST allow resubmission after failed/errored attempts without waiting

#### Competition Timeline

- **FR-055**: Registration period MUST run from September 25, 10 AM through November 20, 5 PM
- **FR-056**: Submission period MUST run from October 23, 10 AM through November 20, 5 PM
- **FR-057**: Team merger deadline MUST be November 6, 5 PM
- **FR-058**: Model verification period MUST run November 21 through November 26
- **FR-059**: Award announcement MUST occur November 27, 5 PM
- **FR-060**: Award ceremony MUST be scheduled for December 3 at Osong Convention Center (subject to change)
- **FR-061**: Award ceremony attendance MUST be required for grand prize recipients

#### Model Constraints and Rules

- **FR-062**: Participants MUST submit only single models (no ensemble voting or parallel model execution)
- **FR-063**: System MUST allow separate models for images and videos within a single submission
- **FR-064**: Face detection or cropping preprocessing MUST be optional (not mandatory)
- **FR-065**: Preprocessing procedures MUST be documented in inference scripts for reproducibility
- **FR-066**: Participants MUST use publicly available data with proper licensing for training
- **FR-067**: Participants MUST accept legal responsibility for copyright, portrait rights, and licensing
- **FR-068**: Award winners MUST disclose model names and license types if using licensed datasets/models
- **FR-069**: Licensed model usage MUST NOT violate individual license terms

#### Competition Keys and Authentication

- **FR-070**: System MUST generate three competition keys per participant (one per CUDA environment)
- **FR-071**: Participants MUST retrieve keys from My Page > Activity History > Competition section
- **FR-072**: Each key MUST correspond to specific CUDA environment using programmatic identifiers: cuda_11_8_key (CUDA 11.8 environment), cuda_12_6_key (CUDA 12.6 environment), cuda_10_2_key (CUDA 10.2 environment)
- **FR-073**: Participants MUST use correct environment key in aifactory.score.submit() function

#### Intellectual Property and Awards

- **FR-074**: Participants MUST retain rights to all submissions
- **FR-075**: Award winners MUST sign technology transfer agreements (copyright and commercialization rights)
- **FR-076**: Organizers MUST NOT acquire rights to non-winning submissions
- **FR-077**: Disputes MUST be resolved through dialogue and negotiation between organizers and participants
- **FR-078**: System MUST distribute total prize pool of 92,000,000 KRW across 7 teams
- **FR-079**: System MUST award grand prize (30,000,000 KRW) to 1st place with Ministry of the Interior and Safety Minister's Award
- **FR-080**: System MUST award excellence prizes (15,000,000 KRW each) to 2nd-3rd place with NFS Director's Award and NIA Director's Award
- **FR-081**: System MUST award merit prizes (8,000,000 KRW each) to 4th-7th place
- **FR-082**: Award winners MUST NOT publicly disclose algorithm details provided to organizers or award status
- **FR-083**: Award winners MUST NOT disclose content that could enable secondary crimes

#### Debugging and Support

- **FR-084**: System MUST display error traceback messages for failed submissions in leaderboard submission history
- **FR-085**: Participants MUST contact organizers via Q&A board for unclear error messages
- **FR-086**: System MUST document common errors: OOM (out of memory) and disk space exceeded
- **FR-087**: System MUST recommend DataLoader worker count ≤ 8 and batch size adjustments for OOM errors
- **FR-088**: System MUST provide contact email cs@aifactory.page for operational support
- **FR-089**: System MUST provide Q&A board at https://aifactory.space/task/9197/qna

#### Data Privacy and Security

- **FR-090**: Dataset files and descriptions MUST be treated as organizer/sponsor assets
- **FR-091**: Participants MUST use provided data only for competition purposes
- **FR-092**: Participants MUST NOT transfer, lend, redistribute, create derivative works, or commercially use provided data
- **FR-093**: Personal information protection MUST be maintained in all inference scripts and preprocessing
- **FR-094**: Copyright and plagiarism MUST comply with relevant laws and regulations

### Key Entities

- **Participant**: Individual registered for competition; attributes include name, nationality, email, registration date, Korean citizenship status, minor status, team membership
- **Team**: Group of up to 5 participants; attributes include team name, leader, members list, formation date, submission count, total daily quota (3), merger status
- **Submission**: Model package uploaded for evaluation; attributes include submission ID, task.ipynb file, timestamp, CUDA environment, status (pending/running/completed/failed), inference duration, error log
- **Model**: Trained deepfake detector packaged in submission; attributes include model files, dependencies, preprocessing scripts, inference code, resource requirements
- **Test Data**: Private evaluation dataset; attributes include filename, file type (image/video), format (JPG/PNG/MP4), duration (for videos), ground truth label (Real/Fake), face count (always 1)
- **Evaluation Result**: Scored submission output; attributes include submission ID, submission.csv file, answer.csv file, Macro F1-score, precision, recall, True Positives, False Positives, True Negatives, False Negatives, leaderboard rank
- **Competition Key**: Authentication token for submission; attributes include key value, participant ID, CUDA version (10.2/11.8/12.6), environment details, generation date
- **Sample Dataset**: Publicly available reference data; attributes include 7 fake images, 5 fake videos, download URL, file formats, purpose (participant guidance)
- **Award**: Prize and recognition for top performers; attributes include rank (1st-7th), prize amount, recipient team, award title, ceremony date, intellectual property transfer status

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Participants can download sample dataset (12 files) in under 5 minutes
- **SC-002**: Participants can successfully submit models through aifactory library with clear error messaging for 95% of submission attempts
- **SC-003**: Automated inference system processes submissions within 3 hours with 99% uptime during competition period
- **SC-004**: Macro F1-score calculation completes within 5 minutes after inference finishes
- **SC-005**: Leaderboard updates display new scores within 10 minutes of evaluation completion
- **SC-006**: System supports 1000+ registered participants and 100+ teams without performance degradation
- **SC-007**: Daily submission quota enforcement prevents any team from exceeding 3 submissions per day
- **SC-008**: Score reproducibility verification succeeds for 95% of top-ranked submissions during November 21-26 period
- **SC-009**: Error traceback messages are accessible for 100% of failed submissions within submission history
- **SC-010**: 90% of participants successfully retrieve and use correct CUDA environment keys on first attempt
- **SC-011**: Team formation and member management operations complete in under 30 seconds
- **SC-012**: Competition timeline milestones (registration, submission deadline, verification, awards) occur on scheduled dates with less than 1 hour deviation
- **SC-013**: Award ceremony logistics support all 7 winning teams with confirmed attendance
- **SC-014**: Intellectual property transfer agreements are signed by 100% of award winners within 2 weeks of announcement
- **SC-015**: Zero unauthorized data redistribution incidents occur during and after competition
- **SC-016**: 90% of participants understand submission format requirements without requiring support
- **SC-017**: Support response time via Q&A board and email averages under 24 hours
- **SC-018**: System prevents ensemble model submissions with 100% accuracy through validation
- **SC-019**: Network isolation after library installation prevents 100% of unauthorized external data access during inference
- **SC-020**: Storage and compute resources accommodate submissions from all teams without infrastructure failures
