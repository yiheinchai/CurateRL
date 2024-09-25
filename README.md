# CurateRL
Reinforcement learning for data curation.

We can consider data curation via reinforcement learning, and AI model that not only learns, but learns which data to learn from based on past reward (reinforcement learning).

Actor, critic and inference model, the critic forms an internal model of predictive the future reward of the data point. the data point reward is derived from the inference loss of the model on the data point. once the critic is sufficiently trained, based on the expected future reward, it can tell the actor whether to learn from the data point or not. if the actor decides to learn from the data point, then the inference model processes the data point and calculates the loss and updates the weight via back prop. the loss then further updates the critic.
