# Team 5 Progress Journal

## Team Meeting Notes

### Meeting 1 - January 21st

- First focus for us will be to work towards the **project proposal**

  - We have 5 weeks of work until this is due
  - The first step towards this will be to deeply understand the supporting material provided

- **For next week**, each team member must:

  - Be completely familiar with the info on the [Jersey Number Recognition Page](https://eval.ai/web/challenges/challenge-page/1952/overview) (SoccerNet Challenge 2023)
  - Review the following papers:
    - [A General Framework for Jersey Number Recognition in Sports Video](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf)
    - [Jersey Number Recognition using Keyframe Identification from Low-Resolution Broadcast Videos](https://arxiv.org/pdf/2309.06285v1)
  - Take notes on your own understanding of each section's architecture, to discuss in our next meeting.
    - Note any possible improvements to each section
    - Each team member should have a good understanding of the foundations of these papers, so that we can share with each other and plan how we will reproduce the results

- **In next week's meeting**, we should:

  - Share our findings on the papers, and ensure we all have a solid understanding
  - Create a plan and action items in a GitHub Project for the following 4 weeks, where we can continue to track to-do's
  - Plan how we will reproduce the paper results

### Meeting 2 - January 28th

#### Key Points Discussed:

- Analyzed interpretations of the two suggested papers, identifying key models and tools used in their pipelines.
- Proposed potential improvements to the pipelines, including:
  - Integrating **DynoV2** and **Clip4Str**.
  - Expanding the dataset to increase domain coverage (considering manual data collection and license plate datasets).
  - Modifying the **ResNet-based legibility classifier** by increasing the number of layers.
  - Exploring different **ViT models**.
  - Experimenting with flipping jersey numbers for data augmentation.

#### Tasks for Next Week:

- Clone and set up the repository from _A General Framework for Jersey Number Recognition in Sports Video_.
- Run and replicate the provided model.
- Explore and analyze the code to understand its structure and functionality.

#### Agenda for Next Meeting:

- Continue discussions on potential improvements.
- Share insights and challenges encountered while running the model locally.

#### Questions for the Professor/TA:

- Is it permissible to use externally collected data to fine-tune the model?
- How significant is efficiency in evaluating the improvement score?

### Meeting 3 - February 5th

#### Key Points Discussed:

- Encountered challenges in setting up the repository on local machines; extended the setup deadline by half a week.

#### Tasks for Next Week:

- Complete the setup by February 9.
- Review the codebase to gain a deeper understanding of the pipeline.

#### Agenda for Next Meeting:

- Share insights about the pipeline.
- Begin discussions on potential concrete improvements.

#### Questions for the Professor/TA:

- None

### Meeting 4 - February 12th, 2025

#### Key Points Discussed:

- Riley successfully executed the pipeline available at https://github.com/mkoshkina/jersey-number-pipeline, with the test phase taking over 5 hours and the training phase over 1 hour. Given these durations, we aim to meticulously plan and evaluate options before modifying the code, as time is a critical factor.

- Team members are reviewing additional research papers and have proposed several potential enhancements, including:

  - Implementing ensemble methods.
  - Training with an expanded dataset featuring fonts specific to soccer jerseys and employing data augmentation techniques (e.g., image distortion and combination).
  - Replacing ResNet34 with a more advanced model, such as ResNet101, for the legibility classifier.
  - The suggestion to allocate tasks based on pipeline components was discussed and may be adopted once all members have a clearer understanding of potential improvements.

Aiden proposed enhancing the legibility classifier to filter out more images earlier in the pipeline, thereby reducing resource consumption.

#### Tasks for Next Week:

- Continue researching and reviewing relevant papers.
- Propose new improvements or refine existing ones.

#### Agenda for Next Meeting:

- Discuss proposed improvements.
- Potentially assign tasks based on the discussed improvements.

#### Questions for the Professor/TA:

- What specific elements are expected in the Progress Journal?
- Could you clarify the requirement "Replicated results of current pipelines and state of the art" for the Project Proposal deliverables?

### Meeting 5 - February 25th, 2025

#### Key Points Discussed:

- We refined and organized ideas for enhancing our deep learning project’s pipeline:
  - Aidan will work on creating a synthetic jersey number dataset to fine-tune the STR system on more jersey-like number fonts.
  - Riley confirmed that he will modify the ResNet architecture and explore a PARSeq alternative.
  - Jerry will focus on pre-training STR models utilizing the CLIP text encoder.
  - Adams will concentrate on adopting an ensemble method to boost recognition accuracy and replace Centroid-ReID with DeepSORT to improve occlusion handling and identity tracking.
- We also began composing the project proposal.

### Tasks for Next Week:

- We aim to get things started on our respective tasks, though we're not sure how far we'll get by then.

### Agenda for Next Meeting:

- Discuss progress on our initial attempt at improving the pipeline.
- Address any challenges encountered and help each other where needed.
-

## Meeting 6 - March 10th

### Key Points Discussed:

- Riley tried Resnet101 for the legibility classifier but it took wayyyyyy too long
- Aidan working on the synthetic image generation
- Jerry working on implementing CLIP
- Somebody else using best GPU at BCCancer so not able to use best compute power available

## Meeting 7 - March 17th

### Key Points

- Aidan finished his algorithim to generate images. It generates with randomness in color, angle, shear, and size of numbers on jersey. Adds random background based on image net
- Riley tried 64bit floating point percision training. It took a lot longer to train, but gave suprisingly very good results, imrpoving a full %point.

### Agenda for next meeting

- Start using syntehtic dataset in training

## Meeting 8 - March 24nd

### Key Points

- Figured out how to train parseq on new data
- tested synthetic images in very small dataset. Found it performed worse. Adjusted generation to add brightness and color randomness, and generated 100k images rather than ~2k
- Noticed that some of the dataset is mislabled, definately interesting to look through. Some outputs were -1 when jersey was very clearly visible
- Found an issue with the prediciton consolidation if only one unique prediction it produces -1 rather than the prediction

### Agenda

- Get more trial runs of different ideas and models

## Meeting 9 - March 30th

### Key Points

- Finishing slideshow for presentation. Trying to get one final dataset for last attempt at image generation. The 100k dataset performed better, but still marginally worse than standard pipeline.
- Training for ResNet50, still training but not converging like we'd hoped. Need more epochs to confirm.

### Agenda for next meeting

- Discuss using syntehtic jerseys on resnet. Triaging poorly generated images manually and use them for legibility classifying. Won't be ready for presentation but maybe for report
- Get more images/graphs to use
- Creating a live demo via creating a one image inference pipeline
