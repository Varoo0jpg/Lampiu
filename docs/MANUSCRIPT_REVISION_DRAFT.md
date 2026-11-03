# ARGUS — Manuscript Revision Draft (I2ASPO, Maximum 13 Pages)

> **Revision basis and scope**
>
> This revision preserves the original manuscript's chapter order, section structure, research framing, and level of explanation wherever these remain consistent with the current ARGUS implementation. Changes are limited to correcting mismatches between the manuscript and the implemented system, aligning the number of research questions and objectives, and distinguishing literature-based PCU values from ARGUS configuration weights.
>
> The 13-page limit applies to the formatted manuscript body. The cover, table of contents, and bibliography are excluded according to the competition rules. This Markdown file does not have fixed pagination; verify the final count in the official I2ASPO template. Chapter 4/results is intentionally not included or revised.
>
> **Indicative page allocation:** Abstract 0.5 page; Chapter 1 2.5 pages; Chapter 2 3.5 pages; Chapter 3 6.5 pages. This is a planning target, not a verified page count.

---

# ABSTRACT

Rapid urbanization in D.I. Yogyakarta has increased pressure on urban mobility, while fixed-time traffic controllers cannot directly adapt to changing traffic demand or differences among vehicle classes. Conventional signal control also provides limited support for identifying and prioritizing emergency vehicles. To address these issues, this study develops ARGUS (Adaptive Real-time Green-time Update System), a software-based traffic-signal simulation for a four-approach intersection. The visual subsystem uses a trained YOLO26n model to detect and classify eight vehicle classes: ambulance, bus, car, fire engine, military vehicle, motorcycle, police car, and truck. Detected vehicle counts are processed using configured vehicle weights to calculate an approach-level traffic score and allocate bounded green-light durations. The emergency-verification subsystem combines visual emergency-vehicle detection with audio classification. Audio is extracted from the soundtrack of a user-selected local video, processed in three-second mono segments at 16 kHz, represented by pretrained YAMNet embeddings, and classified by a Logistic Regression model into ambulance, firetruck, police, and traffic categories. An emergency-priority request is made when the visual emergency confidence and aggregate audio emergency score exceed their configured thresholds on the same approach. The study describes the system architecture, software implementation, and procedures for evaluating vehicle detection, weighted timing allocation, and dual-modal emergency verification. ARGUS currently operates as a software simulation and is not connected to physical traffic-signal hardware; therefore, field-level reductions in congestion, vehicle delay, or emergency travel time are not claimed.

**Keywords:** Adaptive Traffic Control, Computer Vision, Emergency Vehicle Detection, Vehicle Weighting, Intelligent Transportation Systems.

---

# CHAPTER 1 — INTRODUCTION

## 1.1 Background of the Study

Rapid urbanization and escalating vehicular density pose critical challenges to modern urban mobility, directly undermining Sustainable Development Goal 11 (SDG 11: Sustainable Cities and Communities) by straining physical road infrastructure. Inefficient traffic signal control can further increase vehicle idling time and fuel consumption, which is relevant to SDG 13 (Climate Action). Moreover, conventional pre-timed traffic signals do not actively prioritize emergency response vehicles, potentially causing transit delays that compromise public safety and emergency preparedness under SDG 3 (Good Health and Well-being).

In developing urban centers such as the Special Region of Yogyakarta, Indonesia, rapid vehicular growth has increased pressure on a largely fixed road network. The original study background reports more than 3.5 million registered vehicles in 2025; this figure and its definition should remain supported by the corresponding local statistical reference [6]. Yogyakarta's traffic landscape is heterogeneous, with motorcycles, private passenger cars, public transportation, and freight vehicles sharing road space. Conventional pre-timed traffic controllers cannot respond directly to fluctuating traffic volumes. As a result, an approach with relatively low demand may receive green time while another approach experiences longer queues. Uniform vehicle counting also overlooks differences in the road-space requirements of vehicle classes, such as motorcycles and heavy trucks.

These limitations are intensified by the difficulty of identifying emergency vehicles in mixed traffic. Visual detection may be affected by occlusion, while audio detection may be affected by background noise or sounds that resemble sirens. Research on audio-based and audio-visual emergency-vehicle detection provides a basis for combining different sources of evidence [8–12]. A dual-modal approach can therefore be used as a verification condition before the simulation requests emergency priority.

To address these challenges, this study develops an Intelligent Transportation System framework named ARGUS (Adaptive Real-time Green-time Update System). ARGUS uses a trained YOLO26n object-detection model to classify vehicles from video sources representing four intersection approaches. The detected class counts are processed using vehicle weights to calculate approach-level traffic scores and allocate bounded green-light durations. In addition, the system combines visual emergency-vehicle detection with audio classification using pretrained YAMNet embeddings and a Logistic Regression classifier. Emergency priority is requested when both modalities meet their configured thresholds on the same approach.

ARGUS is currently a software-based traffic-signal simulation. It does not control physical traffic lights or establish field-level traffic improvements. This research focuses on implementing and evaluating the vehicle-detection, weighted signal-allocation, and dual-modal emergency-verification functions in controlled scenarios.

## 1.2 Problem Statement

This study addresses the following research questions:

1. How can a vehicle-weighting model based on the Passenger Car Unit (PCU) concept be applied to heterogeneous traffic to calculate approach-level traffic demand and allocate green time under normal operating conditions?
2. How does the signal-transition mechanism operate when the simulation grants emergency priority and subsequently returns to normal adaptive operation?
3. How does the integration of visual vehicle detection and audio classification perform in verifying emergency vehicles, and how does the resulting priority request affect green-time allocation among the four approaches in the software simulation?

## 1.3 Objectives of the Research

The objectives of this study correspond directly to the research questions:

1. To implement a PCU-concept-based vehicle-weighting calculation for heterogeneous traffic and use the resulting approach-level scores to allocate bounded green-light durations.
2. To implement and evaluate the signal transitions during emergency-priority requests and the return to normal adaptive operation.
3. To integrate visual vehicle detection and audio classification for emergency-vehicle verification, and evaluate its effect on simulated green-time allocation among the four approaches.

## 1.4 Significance of the Research

### 1.4.1 Theoretical Significance

1. **Advancement in Heterogeneous Traffic Modeling:** This study implements an algorithmic approach that accounts for multiple vehicle classes, including differences between motorcycles, passenger cars, buses, and trucks, rather than relying solely on uniform vehicle counts.
2. **Integration of Computer Vision and Emergency-Priority Logic:** This study combines visual object detection using YOLO26n with audio classification to establish a dual-modal emergency-verification condition.
3. **Framework for Software-Based Adaptive Traffic Control:** This study provides an implementation framework for combining multi-class vehicle detection, weighted traffic estimation, and adaptive signal timing in a software simulation.

### 1.4.2 Practical and Applied Significance

1. **Evaluation of Adaptive Signal Allocation:** The simulation provides a controlled environment for observing how weighted vehicle counts affect green-time allocation across four approaches.
2. **Evaluation of Emergency-Vehicle Verification:** The dual-modal mechanism provides a basis for examining emergency-verification decisions when visual and audio evidence are considered together.
3. **Basis for Further Development:** The software prototype can support subsequent research on local calibration, hardware integration, and field testing. Such deployment requires additional safety engineering and validation and is not claimed as an outcome of the current study.

## 1.5 Research Hypotheses

The primary detection hypothesis is formulated as follows:

1. **H0 (Null Hypothesis):** Under the defined test conditions, the YOLO26n model does not meet the predetermined vehicle-detection performance criterion.
2. **Ha (Alternative Hypothesis):** Under the defined test conditions, the YOLO26n model meets the predetermined vehicle-detection performance criterion.

The selected detection metric, acceptance criterion, test dataset, and evaluation procedure must be specified before statistical analysis. The hypothesis should not be interpreted as proof of reduced traffic delay or emergency travel time.

---

# CHAPTER 2 — LITERATURE REVIEW

## 2.1 Theoretical Framework

### 2.1.1 Traffic Flow Dynamics and Heterogeneous Characteristics

Traffic flow at signalized intersections is commonly described through volume or flow, speed, and density [5]. In urban areas such as D.I. Yogyakarta, increasing vehicle numbers and limited road capacity can contribute to congestion [6]. Local traffic is heterogeneous and includes a substantial proportion of motorcycles. Consequently, an unweighted vehicle count does not represent the differences in road-space requirements among vehicle classes [1].

The Passenger Car Unit (PCU) concept converts different vehicle classes into an equivalent number of passenger cars [7]. PCU values depend on traffic composition, road geometry, and operating conditions; they should therefore be treated as context-dependent reference values rather than universal constants [7, 16, 22]. For the theoretical basis of this study, the simplified reference categories are presented in Table 2.1.

**Table 2.1. Passenger Car Unit (PCU) equivalent factors for heterogeneous traffic**

| Vehicle classification | PCU value | Reference basis |
|---|---:|---|
| Motorcycle (MC) | 0.25 / 0.33 | MKJI / Sugiarto et al. [7] (simplified) |
| Light Vehicle / Passenger Car (LV) | 1.00 | Passenger-car baseline [7] |
| Medium Heavy Vehicle / Bus (MHV) | 2.00 | MKJI / Sugiarto et al. [7] (simplified) |
| Heavy Vehicle / Large Truck (HV) | 2.00 | MKJI / Sugiarto et al. [7] (simplified) |

The values in Table 2.1 summarize the reference categories used in the original manuscript. They are not a claim that every vehicle subtype has an identical PCU value under all conditions. The operational weights used by ARGUS for its eight detection classes are specified separately in Chapter 3.

The weighted traffic demand on approach (i) is calculated as:

\[
W_i = \sum_{c=1}^{C} N_{i,c}w_c
\]

where (N_{i,c}) is the number of detected vehicles in class (c) on approach (i), and (w_c) is the weight assigned to that class. This calculation provides the traffic score used by the signal-allocation algorithm. It is an estimate derived from detected counts and configured weights, not a direct measurement of physical queue length or road density [3, 4].

### 2.1.2 Computer Vision

Computer vision enables Intelligent Transportation Systems (ITS) to extract and analyse visual information from roadside cameras [5]. For vehicle detection and counting, ARGUS implements a trained YOLO26n model with eight configured vehicle classes. The model returns vehicle categories, bounding boxes, and confidence scores from video frames. The resulting detections are aggregated by approach and used as inputs to the traffic-load estimation module.

Detection results depend on camera position, image resolution, lighting, occlusion, and the similarity between operational video and the training data. Therefore, the model's performance must be evaluated using labelled test data. The confidence threshold and non-maximum suppression (NMS) settings are implementation parameters and should be reported with the experimental configuration.

### 2.1.3 Acoustic Signals

Acoustic signals are pressure waves characterized by frequency and amplitude. Emergency-vehicle sirens can provide supplementary evidence when the vehicle is partially or fully occluded in the visual input. Previous studies have investigated acoustic emergency-vehicle detection and the integration of audio and visual information [8–12].

The active ARGUS audio subsystem uses pretrained YAMNet to generate 1,024-dimensional audio embeddings and a Logistic Regression classifier to identify four categories: ambulance, firetruck, police, and traffic. The emergency audio score is obtained by combining the classifier outputs for ambulance, firetruck, and police. In the current implementation, audio is extracted from the soundtrack of a user-selected local video and processed in three-second mono segments at 16 kHz. MFCC/librosa is not an active parallel detection branch in the current GUI implementation.

### 2.1.4 Control Algorithm

A traffic-control algorithm converts observed traffic conditions into signal-timing decisions. Fixed-time controllers follow predetermined timing plans, whereas adaptive controllers adjust green-light duration in response to traffic demand [5]. Proportional allocation and bounded green-time methods have been discussed in previous adaptive traffic-control studies [2–4].

ARGUS calculates an approach-level weighted traffic score using the vehicle counts and configured weights described in Section 2.1.1. When the sum of all approach scores is greater than zero, the proportional allocation principle is:

\[
G_i^* = G_{total}\frac{W_i}{\sum_{j=1}^{4}W_j}
\]

The resulting allocation is subject to the configured minimum and maximum green-time limits. The current ARGUS configuration uses a total green-time budget of 120 seconds, with a minimum of 10 seconds and a maximum of 40 seconds for each approach. The implementation applies bounded redistribution; therefore, the equation expresses the proportional allocation principle and is not, by itself, the complete bounded algorithm. When all approach scores are zero, the simulation uses equal allocation.

Yellow and all-red intervals are configured as 3 seconds and 1 second, respectively, and are additional to the green-time budget. Emergency priority is requested when the visual emergency confidence and aggregate audio emergency score both exceed 0.65 on the same approach. The configured threshold comparison is strict. The simulation retains the yellow and all-red transition intervals when changing signal phases. These are software parameters and should not be presented as field-calibrated safety settings.

---

# CHAPTER 3 — RESEARCH METHOD

## 3.1 System Requirements Analysis and System Characteristics

ARGUS (Adaptive Real-time Green-time Update System) is a computer-vision-based software simulation developed to adjust traffic-signal timing according to observed traffic conditions. The system uses YOLO26n for vehicle detection from multiple video sources, vehicle weighting to estimate approach-level traffic demand, and an emergency-priority mechanism based on combined visual and audio classification. Previous studies have examined computer-vision-based vehicle detection and adaptive traffic-signal control [1–4].

### 3.1.1 System Requirements Analysis

ARGUS requires video sources representing the four approaches of a four-leg intersection: North, East, South, and West. The sources may include configured CCTV streams, webcams, built-in camera sources, or local video files, depending on the test setup. The YOLO26n model detects and classifies vehicles in the video frames. The resulting class counts are processed using the configured vehicle weights to estimate traffic demand for each approach [2, 7].

The system also requires an audio-classification mechanism for emergency-vehicle verification. In the current GUI, audio is obtained from the soundtrack of a selected local video and processed using pretrained YAMNet embeddings and a Logistic Regression classifier. Additional requirements include real-time or near-real-time processing, software compatibility, operational stability, and a graphical interface that presents vehicle detections, traffic scores, audio scores, and signal states.

ARGUS is currently a software simulation. Physical traffic-light hardware and IoT-based signal control are outside the implemented system scope.

### 3.1.2 System Characteristics

ARGUS is characterized by multi-source vehicle detection using YOLO26n and the representation of a four-approach intersection. The system identifies eight vehicle classes: ambulance, bus, car, fire engine, military vehicle, motorcycle, police car, and truck. Vehicle counts are processed using configured operational weights to estimate traffic demand for each approach.

The weighted scores are used to allocate green-light durations subject to minimum and maximum limits. In addition, ARGUS incorporates dual-modal emergency verification by combining visual detection of emergency-vehicle classes with audio classification from a selected local video's soundtrack. The GUI displays the video sources, vehicle counts, traffic scores, audio-classification scores, and simulated signal states.

## 3.2 System Architecture and Algorithm Design

### 3.2.1 Input

The system receives two types of input: visual and audio. Visual input consists of video streams or video files representing the North, East, South, and West approaches. YOLO26n processes the video frames to detect and classify eight vehicle classes, including ambulances, fire engines, and police cars.

Audio input in the current GUI is obtained from the soundtrack of a user-selected local video. The audio is processed in three-second mono segments at a sampling rate of 16 kHz. Pretrained YAMNet generates audio embeddings, which are classified by Logistic Regression. Thus, the current implementation does not use a separate MP3 upload or a live microphone as its active audio input.

The quality of both inputs affects the analysis. Visual detection depends on video quality, camera angle, lighting, occlusion, and model performance. Audio classification may be affected by background noise, soundtrack quality, and similarity between sirens and other sounds. Classifier scores are model outputs and should not be interpreted as calibrated probabilities unless a calibration procedure has been conducted.

### 3.2.2 Process

The system processes visual and audio data through connected modules. First, YOLO26n detects and classifies vehicles in the video frames. The detected class counts are aggregated by approach and smoothed using a five-frame median filter. Each class count is multiplied by its configured operational weight, and the weighted values are summed to obtain the traffic score for each approach.

The approach-level scores are transferred to the signal-timing module. The module distributes a total green-time budget of 120 seconds among the four approaches, subject to a minimum of 10 seconds and a maximum of 40 seconds per approach. When all approach scores are zero, the system uses equal allocation.

For emergency verification, the soundtrack of the selected local video is divided into three-second segments, converted to mono, and resampled to 16 kHz. YAMNet generates a 1,024-dimensional embedding for each segment, and Logistic Regression classifies the embedding into ambulance, firetruck, police, or traffic. The emergency audio score is calculated from the ambulance, firetruck, and police outputs.

The emergency-verification module checks the visual emergency confidence and aggregate audio emergency score. An emergency-priority request is made only when both scores exceed 0.65 on the same approach. Otherwise, the simulation continues using the weighted traffic-allocation process. Yellow and all-red intervals are retained during signal transitions.

### 3.2.3 Output

The graphical interface is designed to allow the operator to monitor traffic conditions and observe the adaptive signal simulation. It displays video from the four approaches—North, East, South, and West—together with vehicle detections and class counts. The interface also presents weighted traffic scores, simulated red, yellow, and green signal states, countdown timers, and audio-classification information.

The GUI provides controls for selecting video sources and operating the simulation. System observations can be recorded and exported in CSV format for subsequent analysis. These records are periodic system snapshots and do not represent unique-vehicle trajectories or cross-camera vehicle identities. At the current development stage, physical traffic-signal control through IoT hardware has not been integrated.

## 3.3 Software Implementation and Module Integration

### 3.3.1 Software Implementation

#### 3.3.1.1 YOLO26n-Based Vehicle Detection

Vehicle detection is implemented using a trained YOLO26n model integrated through the Ultralytics library and OpenCV. The model identifies eight vehicle classes: ambulance, bus, car, fire engine, military vehicle, motorcycle, police car, and truck. Video frames from four approaches (North, East, South, and West) are processed using a default confidence threshold of 0.15 and an NMS IoU threshold of 0.60. Detected vehicles are classified and counted according to their categories. A five-frame median filter is applied to reduce short-term fluctuations in detection counts before the results are transferred to the traffic-load estimation module.

The final research report should identify the model checkpoint and its training dataset. Training and independent test metrics should only be reported when supported by the corresponding training and evaluation records.

#### 3.3.1.2 YAMNet-Based Audio Classification and Siren Detection

Audio classification is implemented using pretrained YAMNet embeddings combined with a Logistic Regression classifier. Audio is extracted from the soundtrack of a user-selected local video and processed in three-second mono segments at a sampling rate of 16 kHz. YAMNet generates 1,024-dimensional audio embeddings, which are classified into four categories: ambulance, firetruck, police, and traffic. The emergency score is calculated by combining the classifier outputs for the three emergency-vehicle categories. The resulting score is displayed through the Audio Confidence panel and used for emergency-vehicle verification.

#### 3.3.1.3 PCU-Based Traffic-Load Estimation and Signal Control

The theoretical PCU reference categories are presented in Table 2.1. For implementation, ARGUS maps its eight detected vehicle classes to the operational weights shown in Table 3.1. These values are the current simulator parameters; they should not be interpreted as independently validated PCU standards for every vehicle subtype.

**Table 3.1. Operational vehicle weights used in ARGUS**

| Detected vehicle class | Operational weight |
|---|---:|
| Ambulance | 1.00 |
| Bus | 2.00 |
| Car | 1.00 |
| Fire engine | 2.00 |
| Military vehicle | 2.00 |
| Motorcycle | 0.33 |
| Police car | 1.00 |
| Truck | 2.00 |

Traffic demand is estimated by multiplying the detected count of each class by its corresponding operational weight. The weighted values are summed to obtain the traffic score for each approach. These scores determine the bounded allocation of green-light duration within a total green-time budget of 120 seconds. Each approach receives a minimum of 10 seconds and a maximum of 40 seconds. When no vehicles are detected on any approach, the green-time budget is divided equally among the four approaches. Signal transitions consist of a 3-second yellow interval and a 1-second all-red interval. Emergency prioritization is requested when visual and audio scores on the same approach both exceed their configured threshold of 0.65.

#### 3.3.1.4 Graphical User Interface and Research Data Logging

The graphical user interface is developed using CustomTkinter to display video feeds, vehicle counts, weighted traffic scores, traffic-signal states, countdown timers, and audio-classification scores. It also provides simulation controls and video-source selection. The research data-logging function records system observations, including vehicle classifications, weighted traffic scores, signal phases, and active approaches. Recorded data can be exported in CSV format for further analysis. The exported records represent system snapshots and should not be described as unique-vehicle tracking data.

### 3.3.2 Module Integration

The ARGUS modules are integrated through two connected processing pipelines: visual vehicle detection and audio classification. The visual pipeline processes video frames using YOLO26n to obtain vehicle classifications and counts. These results are transferred to the traffic-load estimation module, which applies the operational vehicle weights and calculates the traffic score for each approach.

The audio pipeline processes the soundtrack of a selected local video using YAMNet embeddings and Logistic Regression to generate audio-classification scores. The visual and audio outputs are connected through a dual-modal verification mechanism. When the visual emergency confidence and aggregate audio emergency score both exceed their configured thresholds on the same approach, the system requests emergency priority for that approach. The signal transition retains the configured yellow and all-red intervals before the next green phase. Otherwise, the simulation continues using weighted traffic-based signal allocation.

Processing results are displayed through the graphical interface. Separate execution threads are used for camera processing and audio analysis to maintain interface responsiveness. Research data are recorded periodically and can be exported in CSV format. At the current development stage, ARGUS operates as a software-based traffic simulation and is not connected to physical traffic-signal controllers.

## 3.4 Experimental Setup and Testing Scenarios

The following subsections describe the testing procedures to be conducted and documented. They must not be written as completed experimental findings unless the corresponding tests and records are available.

### 3.4.1 Simulation Environment and Configuration

The simulation represents a four-leg intersection with North, East, South, and West approaches. Video sources may consist of configured CCTV streams, webcams, built-in camera sources, or local video files. The experimental report should identify the source assigned to each approach, the computer specifications, operating system, Python environment, package versions, YOLO26n checkpoint, and relevant GUI settings.

For audio evaluation, the selected local video and its soundtrack must be identified. The current GUI uses the soundtrack of a selected local video rather than a separate MP3 input or live microphone. When multiple local videos are selected, the active GUI audio source is the first local video found in approach order; this limitation must be considered when defining the test setup.

### 3.4.2 Heterogeneous Traffic Density and Vehicle-Weighting Protocol

Traffic scenarios should represent low, medium, and high vehicle demand using known vehicle counts and class compositions. The test data should include motorcycles, cars, buses, and trucks, together with the additional vehicle classes supported by the detector where relevant. For each scenario, calculate the expected weighted score for every approach using the operational weights in Table 3.1.

Record the four approach scores and the green durations assigned by ARGUS. Verify that each green duration remains between 10 and 40 seconds and that the four green allocations sum to 120 seconds. The zero-demand condition should also be tested to verify equal allocation. These tests evaluate the timing calculation and its configured constraints; they do not by themselves establish reduced physical queue lengths or traffic delay.

### 3.4.3 Dual-Modal Emergency-Vehicle Verification Protocol

The emergency-verification evaluation should include ambulance, fire-engine, and police-vehicle cases. Positive and negative scenarios should be prepared for the visual and audio inputs. The test set should include ordinary traffic audio and cases in which one modality is absent or its score remains below the configured threshold.

For each test, record the visual emergency confidence, audio emergency score, approach mapping, verification decision, and decision time. A positive dual-modal verification requires both scores to exceed 0.65 on the same approach. The evaluation should distinguish successful verification from cases where only one modality indicates an emergency. Claims that dual-modal detection outperforms single-modal detection require a direct comparison using the same test scenarios.

### 3.4.4 Signal Transition and Return-to-Normal Validation

Record the signal state and transition durations before, during, and after an emergency-priority request. Verify that the configured yellow and all-red intervals are retained. Also record which approach receives the next green phase and how the simulation returns to normal weighted allocation.

A time-compensation algorithm or recovery mechanism should only be described as implemented if it is present in the current software and verified through testing. The current manuscript does not assume that delayed approaches receive additional compensatory green time or that traffic equilibrium is restored within a measured number of cycles.

### 3.4.5 Environmental Robustness and Signal-Degradation Testing

If environmental robustness is evaluated, prepare separate test conditions for lighting variation, camera angle, partial occlusion, image quality, and audio background noise. Record the test conditions and the corresponding detection or classification results. Audio tests may include ordinary urban sounds that could interfere with siren classification.

Robustness under day, night, rainy, occluded, or noisy conditions must not be claimed unless those conditions are represented in the test data and the results are reported. This subsection defines possible evaluation conditions; it does not state that all such tests have already been completed.

## 3.5 Performance Metrics and System Evaluation

### 3.5.1 Computer Vision and Object-Detection Metrics

YOLO26n vehicle-detection performance should be evaluated using Precision (P), Recall (R), F1-score, and mean Average Precision (mAP). Report results for each vehicle class and identify the confidence and IoU conventions used in the evaluation. True positives (TP), false positives (FP), and false negatives (FN) must be determined against labelled ground-truth annotations. Where applicable, report mAP50 and mAP50-95.

### 3.5.2 Traffic-Load Estimation and Green-Time Allocation Metrics

Evaluate the traffic-load calculation by comparing the weighted scores produced by ARGUS with scores calculated manually from known vehicle counts and the operational weights in Table 3.1. Evaluate signal allocation by recording each approach's green duration, checking the 10–40-second bounds, and verifying that the total green allocation equals 120 seconds. The zero-demand fallback should be evaluated separately.

Queue length, average intersection delay, throughput, and vehicle waiting time require a traffic-flow model or physical measurements and a defined baseline. They cannot be inferred from vehicle detections and green-time allocations alone.

### 3.5.3 Emergency-Verification Accuracy and Response Time

Dual-modal verification should be evaluated using a confusion matrix, Precision, Recall, F1-score, false-positive rate (FPR), and false-negative rate (FNR). Report the visual and audio scores and whether both thresholds were exceeded on the same approach. Response time should be measured from a clearly defined input event to the verification decision or the corresponding simulated signal transition. The measurement endpoint must be stated consistently.

### 3.5.4 Signal Transition Compliance

Measure the duration of yellow and all-red intervals and compare them with their configured values of 3 seconds and 1 second. Record the signal sequence during emergency-priority requests and the return to normal weighted allocation. A recovery-cycle count, delay-recovery ratio, or claim of restored traffic equilibrium should only be reported if a corresponding recovery mechanism and measurement procedure are implemented and tested.

---

# Author Checks Before Submission

This checklist is an internal editing note and should be removed from the submitted manuscript unless the competition template permits it.

- Verify the source, definition, and year of the reported Yogyakarta vehicle-registration figure in the original reference [6].
- Confirm the PCU categories and values in Table 2.1 against the exact cited standard and reference. The simplified values must not be presented as universal.
- Confirm that Table 3.1 matches the operational weights in the current software.
- Verify the YOLO26n checkpoint, training dataset, data split, and independent test metrics before reporting model performance.
- Verify the audio dataset, class counts, train/validation/test split, and original training outputs before reporting any classifier metrics.
- Confirm that the test setup reflects the current local-video soundtrack selection behaviour.
- Do not claim physical traffic-light control, time compensation, reduced queue length, reduced emergency travel time, or field deployment unless those capabilities are implemented and directly evaluated.
- Check every in-text citation against the bibliography and the specific claim it supports.

**Page-count verification is still required in the official I2ASPO Word template.** If the formatted manuscript exceeds 13 pages, shorten repeated wording and nonessential detail without removing the global-to-local background structure, the PCU reference table, the aligned research questions and objectives, or the methodological steps needed to reproduce the simulation.
