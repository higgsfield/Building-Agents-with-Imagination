# Building Agents with Imagination

<img width="160px" height="22px" href="https://github.com/pytorch/pytorch" src="https://pp.userapi.com/c847120/v847120960/82b4/xGBK9pXAkw8.jpg">

![](https://i.imgur.com/un9gSKe.gif)

Intelligent agents must have the capability to ‘imagine’ and reason about the future. Beyond that they must be able to construct a plan using this knowledge. [[1]](https://deepmind.com/blog/agents-imagine-and-plan/) This tutorial presents a new family of approaches for imagination-based planning:
-  Imagination-Augmented Agents for Deep Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1707.06203)
-  Learning and Querying Fast Generative Models for Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1802.03006)

## The tutorial consists of 4 parts:

#### 1. MiniPacman Environemnt
MiniPacman is played in a 15 × 19 grid-world. Characters, the ghosts and Pacman, move through a maze. The environment was written by [@sracaniere](https://github.com/sracaniere) from DeepMind.<br>
[[minipacman.ipynb]](https://github.com/higgsfield/Building-Agents-with-Imagination/blob/master/1.minipacman.ipynb)

#### 2. Actor Critic
Training standard model-free agent to play MiniPacman with advantage actor-critic (A2C)<br>
[[actor-critic.ipynb]](https://github.com/higgsfield/Building-Agents-with-Imagination/blob/master/2.actor-critic.ipynb)

#### 3. Environment Model
Environment model is a recurrent neural network which can be trained in an unsupervised
fashion from agent trajectories: given a past state and current action, the environment model predicts
the next state and reward.<br>
[[environment-model.ipynb]](https://github.com/higgsfield/Building-Agents-with-Imagination/blob/master/3.environment-model.ipynb)

#### 4. Imagination Augmented Agent [in progress]
The I2A learns to combine information from its model-free and imagination-augmented paths. The environment model is rolled out over multiple time steps into the future, by initializing the imagined trajectory with the present time real observation, and subsequently feeding simulated observations into the model. Then a rollout encoder processes the imagined trajectories as a whole and **learns to interpret it**, i.e. by extracting any information useful for the agent’s decision, or even ignoring it when necessary This allows the agent to benefit from model-based imagination without the pitfalls of conventional model-based planning.<br> 
[[imagination-augmented agent.ipynb]](https://github.com/higgsfield/Building-Agents-with-Imagination/blob/master/4.imagination-augmented%20agent.ipynb)

## More materials on model based + model free RL

  - The Predictron: End-To-End Learning and Planning [[arxiv]](https://arxiv.org/abs/1612.08810) [[https://github.com/zhongwen/predictron]](https://github.com/zhongwen/predictron)
  - Model-Based Planning in Discrete Action Spaces [[arxiv]](https://arxiv.org/abs/1705.07177)
  - Schema Networks: Zero-shot Transfer with a Generative Causal Model of Intuitive Physics [[arxiv]](https://arxiv.org/abs/1706.04317)
  - Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning [[arxiv]](https://arxiv.org/pdf/1803.00101v1.pdf)
  - TEMPORAL DIFFERENCE MODELS: MODEL-FREE DEEP RL FOR MODEL-BASED CONTROL [[arxiv]](https://arxiv.org/pdf/1802.09081v1.pdf) [[https://github.com/vitchyr/rlkit]](https://github.com/vitchyr/rlkit)
  - Universal Planning Networks [[arxiv]](https://arxiv.org/abs/1804.00645)
  - World Models [[arxiv]](https://worldmodels.github.io/) [[https://github.com/AppliedDataSciencePartners/WorldModels]](https://github.com/AppliedDataSciencePartners/WorldModels)
  - Recall Traces: Backtracking Models for Efficient Reinforcement Learning [[arxiv]](https://arxiv.org/pdf/1804.00379.pdf)
  - [[Learning by Playing – Solving Sparse Reward Tasks from Scratch ]](https://arxiv.org/abs/1802.10567)  [[https://zhuanlan.zhihu.com/p/34222231]](https://zhuanlan.zhihu.com/p/34222231)  [[https://github.com/HugoCMU/pySACQ]](https://github.com/HugoCMU/pySACQ)
  - [[Hindsight experience replay]](https://arxiv.org/abs/1707.01495) [[https://github.com/openai/baselines/tree/master/baselines/her]](https://github.com/openai/baselines/tree/master/baselines/her)
  - [[https://github.com/pathak22/zeroshot-imitation]](https://github.com/pathak22/zeroshot-imitation)
  - [[Emergence of Structured Behaviors from Curiosity-Based Intrinsic Motivation]](https://arxiv.org/abs/1802.07461)
  - [[Learning Awareness Models]](https://arxiv.org/abs/1804.06318)
  - ray-project
  
  
  
  -------------------------------------------------------------------------------
  
  
  
  - [[Vector-based navigation using grid-like representations in artificial agents]](https://pan.baidu.com/s/1RUzMKQb95qUf5cv6XRsexA)
  - Learning to Navigate in Cities Without a Map [[arxiv]](https://arxiv.org/abs/1804.00168) [[https://zhuanlan.zhihu.com/p/35319354]](https://zhuanlan.zhihu.com/p/35319354)  
  - [[Emergence of grid-like representations by training recurrent neural networks to perform spatial localization]](https://openreview.net/forum?id=B17JTOe0-)
  - Divide-and-Conquer Reinforcement Learning
  - Meta-Learning and Universality: Deep Representations and Gradient Descent can Approximate any Learning Algorithm
  - DDCO: Discovery of Deep Continuous Options for Robot Learning from Demonstrations ;  Multi-Level Discovery of Deep Options
  - Imagination Machines: A New Challenge for Artificial Intelligence
  - [Sensorimotor Robot Policy Training using Reinforcement Learning](https://www.diva-portal.org/smash/get/diva2:1208897/FULLTEXT01.pdf)
  - Meta learning shared hierarchies (ref by UPN)  https://github.com/openai/mlsh
  - https://github.com/andyzeng/visual-pushing-grasping
  - https://github.com/hoangminhle/hierarchical_IL_RL
  - Parametrized Hierarchical Procedures for Neural Programming  http://roydfox.com/category/publications
  - https://github.com/ray-project/ray/blob/master/examples/carla/a3c_lane_keep.py
  - Diversity is All You Need: Learning Skills without a Reward Function   https://github.com/haarnoja/sac   https://sites.google.com/view/diayn  Soft Actor-Critic  rlkit TDM https://github.com/vitchyr/rlkit
  - https://arxiv.org/abs/1805.07917 Evolutionary Reinforcement Learning Shauharda Khadka, Kagan Tumer (Submitted on 21 May 2018)
  - Disentangling the independently controllable factors of variation by interacting with the world https://arxiv.org/abs/1802.09484  Disentangling Controllable and Uncontrollable Factors of Variation by Interacting with the World https://arxiv.org/abs/1804.06955
  - Hierarchical Reinforcement Learning with Hindsight https://arxiv.org/abs/1805.08180
  - Generalisation of structural knowledge in the Hippocampal-Entorhinal system  https://www.groundai.com/project/generalisation-of-structural-knowledge-in-the-hippocampal-entorhinal-system/
  - Sensorimotor Robot Policy Training using Reinforcement Learning https://www.diva-portal.org/smash/get/diva2:1208897/FULLTEXT01.pdf 
  - ray-project 
  
  
  
  -----------------------------------------------------------------------------------------------
  
  
  
  
  - https://www.groundai.com/project/understanding-disentangling-in-vae/

  - Divide-and-Conquer Reinforcement Learning https://arxiv.org/abs/1711.09874
  -  https://www.groundai.com/project/data-efficient-hierarchical-reinforcement-learning/  
  - https://github.com/openai/glow
  - https://github.com/ml-jku/baselines-rudder
  - https://github.com/ashedwards/ILPO 
  - https://github.com/musyoku/generative-query-network
  - https://github.com/ermongroup/Variational-Ladder-Autoencoder  Learning Hierarchical Features from Generative Models
  - https://www.groundai.com/project/unsupervised-learning-of-latent-physical-properties-using-perception-prediction-networks/ 
  - Learning models for visual 3D localization with implicit mapping  https://arxiv.org/abs/1807.03149
  - Representation Learning with Contrastive Predictive Coding https://arxiv.org/pdf/1807.03748.pdf
  - https://www.groundai.com/project/deep-hidden-physics-models-deep-learning-of-nonlinear-partial-differential-equations/
  - https://www.groundai.com/project/unsupervised-learning-of-latent-physical-properties-using-perception-prediction-networks/
  - Discovering physical concepts with neural networks  https://arxiv.org/pdf/1807.10300.pdf
  - https://www.groundai.com/project/disentangling-by-partitioning-a-representation-learning-framework-for-multimodal-sensory-data/
  - https://www.groundai.com/project/a-deep-generative-model-for-disentangled-representations-of-sequential-data/
  
  
  ----------------------------------------------------------------------------------------------------------------
  
  - https://arxiv.org/abs/1801.04062  MINE: Mutual Information Neural Estimation
  - https://www.groundai.com/project/temporal-difference-variational-auto-encoder/  
  - Hyperspherical Variational Auto-Encoders https://github.com/nicola-decao/s-vae-tf
  - https://www.groundai.com/project/consistent-generative-query-networks/  gqn-video
- Isolating Sources of Disentanglement in Variational Autoencoders https://arxiv.org/pdf/1802.04942.pdf  https://github.com/rtqichen/beta-tcvae
- Replicating "Understanding disentangling in β-VAE"  https://github.com/miyosuda/disentangled_vae
- Unsupervised Learning of Disentangled and Interpretable Representations  from Sequential Data  https://github.com/wnhsu/FactorizedHierarchicalVAE
- Learning deep representations by mutual information estimation and maximization https://arxiv.org/abs/1808.06670  https://github.com/rdevon/DIM
- https://arxiv.org/pdf/1804.02086.pdf  Structured Disentangled Representations 

- Formal Limitations on the Measurement of Mutual Information https://arxiv.org/abs/1811.04251

- FutureGAN: Anticipating the Future Frames of Video Sequences using Spatio-Temporal 3d Convolutions in Progressively Growing Autoencoder GANs  https://arxiv.org/abs/1810.01325

- Learning Latent Dynamics for Planning from Pixels https://arxiv.org/pdf/1811.04551.pdf

- Towards Governing Agent’s Efficacy:Action-Conditional β -VAE for Deep Transparent Reinforcement Learning   https://arxiv.org/pdf/1811.04350.pdf

- Improving Generalization for Abstract Reasoning Tasks Using Disentangled Feature Representations https://arxiv.org/pdf/1811.04784.pdf

- Pay attention! - Robustifying a Deep Visuomotor Policy through Task-Focused Attention https://arxiv.org/abs/1809.10093

- Unsupervised Emergence of Spatial Structure from Sensorimotor Prediction https://arxiv.org/abs/1810.01344

Hybrid Active Inference  https://arxiv.org/abs/1810.02647
    We describe a framework of hybrid cognition by formulating a hybrid cognitive agent that performs hierarchical active inference across a human and a machine part. We suggest that, in addition to enhancing human cognitive functions with an intelligent and adaptive interface, integrated cognitive processing could accelerate emergent properties within artificial intelligence. To establish this, a machine learning part learns to integrate into human cognition by explaining away multi-modal sensory measurements from the environment and physiology simultaneously with the brain signal. With ongoing training, the amount of predictable brain signal increases. This lends the agent the ability to self-supervise on increasingly high levels of cognitive processing in order to further minimize surprise in predicting the brain signal. Furthermore, with increasing level of integration, the access to sensory information about environment and physiology is substituted with access to their representation in the brain. While integrating into a joint embodiment of human and machine, human action and perception are treated as the machine's own. The framework can be implemented with invasive as well as non-invasive sensors for environment, body and brain interfacing. Online and offline training with different machine learning approaches are thinkable. Building on previous research on shared representation learning, we suggest a first implementation leading towards hybrid active inference with non-invasive brain interfacing and state of the art probabilistic deep learning methods. We further discuss how implementation might have effect on the meta-cognitive abilities of the described agent and suggest that with adequate implementation the machine part can continue to execute and build upon the learned cognitive processes autonomously. 
    
 - LEARNING ACTIONABLE REPRESENTATIONS WITH GOAL-CONDITIONED POLICIES
