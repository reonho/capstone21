## Key Points

Human pose estimation is the computer vision task of estimating the configuration (‘the pose’) of the human body by localizing certain key points on a body within a video or a photo. This localization can be used downstream to predict if a person is standing, sitting, lying down, or doing some activity like dancing or jumping.

NVIDIA's BPNET is a fully convolutional model with architecture consisting of

    1. a backbone network (VGG)
    2. an initial estimation stage which does a pixel-wise prediction of confidence maps (heatmaps) and part affinity fields
    3. followed by multistage refinement (0 to N stages) on the initial predictions.

We will explore these steps in the sections below.

### Backbone Network

Given a base image below:

![base](%placeholder1%)

### Confidence Maps and Part Affinity Fields

![heatmap](%placeholder2%)

![paf](%placeholder3%)

### Multistage Refinement

%placeholder4%

![final_pred](%placeholder5%)
