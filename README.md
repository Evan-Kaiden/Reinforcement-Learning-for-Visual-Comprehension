# RECAP 
Reinforcement for Confidence-Aware Perception
# Driving Question
If we give a Reinforcement Learning agent 2 tools: **vision** and **memory** will it be able to scan an image and learn "where to look" 

# Implementation
1. Using the A2C algorithm train a policy that takes in a context vector of the previous state to generate a probabily distribution of "where to look next" 
2. Based on the generated position take a look at the surrounding pixels and generate a embedding 
3. Store this embedding in the Agents memory (A LSTM) and generate the context vector for the next state
4. Repeat (1-3) until the agent decides to stop looking
5. Concat embeddings from t<sub>0</sub>–t<sub>n</sub> and pass through a RNN -> Classifier to get final class predictions
