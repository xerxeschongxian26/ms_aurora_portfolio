# aurora-inference

This repository contains a self-hosted inference service for the open-sourced Aurora model by Microsoft Research. The model was first released in June 2024, followed by its associated Nature article in October 2025. The links to official GitHub and Hugging Face repositories are found below.

[Aurora: A Foundation Model for the Earth System](https://github.com/microsoft/aurora)
[Hugging Face Repository for Aurora](https://huggingface.co/microsoft/aurora)

## Why this project

The project scope is a self-hosted engineering project based on an open-sourced foundation model, focusing on build towards a serving and inference optimisation layer. This is a deliberate bounded *initial* scope that emphasises core engineering skills whilst keeping costs low.

With growing concerns over data privacy, data governance and the overall sovereignty of AI systems, the benefits and ability to self-host one's AI model cannot be overstated. To that end, this project has two goals:

1. Self-hosting an open-sourced foundation model
    The checkpoints/weights of the trained Aurora foundation model are available on Hugging Face. The goal is to develop an end-to-end deep learning pipeline towards serving inference over an API.
2. Optimise for inference and understand trade-offs in performance
    Based on recent issues described on the official repository, there remains possible outstanding optimisations that can be performed on the model. The papers emphasise model architecture details and model fine-tuning whilst the official kit offer optimised options.
    Whilst such optimisation techniques are common across deep learning models, to the best of my knowledge, that there are no documented results on the trade-offs in Aurora's performance associated with these techniques.
    *i.e., how far can inference engineering be pushed before forecast quality suffers*. While learning to apply these techniques, I am to document these trade-offs in this repository.


At time of writing, the value of fine-tuning foundation models is acknowledged but such plans are reserved for the future.

## Current stage: Stage 0

Stage 0 forms the scaffolding.


## Quickstart



## Architecture



## Scope boundary



## Roadmap



## Project Development Notes

### Background
Weather forecasting techniques have been in development since the early 1920s and the availability of compute resources and better modelling of the complex physics governing our Earth's system have improved the accuracy and size of the forecast window.

Modern weather forecasting techniques rely on classical numerical methods, solving the large systems of equations, one small increment at a time, repeated across various initial conditions, in order to arrive at a distribution of forecasts for the Earth system. This computationally expensive number crunching process is made possible only by the availablilty of supercomputing resources.

With the resurgence of deep neural networks and the explosion of computing resources, sufficiently trained deep learning models have made continued progress in demonstrating the ability to replace this computationally expensive step, driving down the cost of forecasting whilst improving forecasts. The Aurora weather foundation model by Microsoft Research is one of the latest in the line of AI weather forecasting models which include GenCast and GraphCast (Google DeepMind), Pangu-Weather (Huawei Cloud), FourCastNet (Nvidia), AIFS (European Centre for Medium-Range Weather Forecast), amongst others.

### Models, Architecture and Mathematical Logic
- The Aurora models appears to consists of smaller sub models for predicting air pollutions (particulate concentration), wave dynamics and of course, weather. Sub categories are also trained at different spacial resolutions
- The core architecture is a Swin Transformer and Perceiver
- The models are all pretrained and are available in a full and small version. The full version requires 40GB of memory, 5GB of which consists of the weights. This is approximately the capacity of an entry level enterprise GPU such as the NVIDIA RTX H100
- The smaller pretrained versions are provided for debugging purposes. It's input and output signatures (i.e. the shape of the heterogenous inputs and outputs) are identical to the full model but differ only in the complexity of the architectures, which consists of less layers and possibly a simpler architecture
- Input Time Step - t-1, and t
- Output time step - t + 1
- The input and output time steps resembles a 2nd order Markov property, despite classical numerical methods assuming a 1st order Markov
Property
### Data, Data Sources and Data Integrity
- Input data are heterogenous and unnormalised when fed into the network. Normalisation occurs inside the model
- The Batch and Metadata data classes are custom data classes that are part of the Aurora package
- Both the input and output are of the custom dataclass type Batch

### Ideas
- Demonstrate the ability to predict the cyclone path (and state?) for the recent Hurrican Melissa (Oct. 2025)

## License

MIT — see [LICENSE](LICENSE).
