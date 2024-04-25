#!/usr/bin/env python
# coding: utf-8

# (Bayes:bayes)=
# # Bayes Filter
# The idea of taking previous knowledge into account is nothing new by any means - it arises from the established field of *Bayesian Statistics*. Since the Kalman Filter essentially is only a realization of the Bayes Filter, it makes sense to start building our intuition based on the generic framework of the Bayes Filter. The Bayes Filter is an recursive estimation technique that estimates the current state of a system given measurements and control commands and it is an online estimation algorithm meaning that it only considers the past state togehter with the most recent measurements and control commands.
# 
# The Bayes Filter is just a framework, and is not used as an actual realization in practice. For that we have various realizations like the, e.g. the Kalman Filter, EKF, UKF and the Particle Filter that are all filters based on this generic framework.
# 
# However, all these realizations will differ in the way they make assumptions about the problem, e.g. the Kalman Filter assumes everything is Gaussian and linear, while the particle Filter can represent arbitrary distrubutions and is also non-linear. However, this flexibility comes at higher computational cost.
# 
# ![navigation](fig/probability.jpg)
# 
# ```{note}
# Bayesian Rule - "What is your belief after seeing the evidence?"
# ```
# 
# $$
# bel(x_t)=\eta p(z_t|x_t) \int p(x_t|u_t, x_{t-1}) bel(x_{t-1})dx_{t-1}
# $$
# 
# The Bayes Filter consists of two models: a *measurement model* and a *motion model*. The measurement model $\eta p(z_t|x_t)$ describes "what's the likelihood of obtaining the measurement $z$ given the current state $x_t$?", while the motion model $p(x_t|u_t, x_{t-1})$ describes "What's the likelihood that the state $x_t$ advances to $x_{t-1}$ given the control commands $u$?". The $\eta$ is the normalization factor.
# 
# Let's start with the simple hallway example from the field of robotics {cite}`thrun2005probabilistic`.
# 
# Assume that the robots world consists of a one dimensional hallway, whose only features are three lights hanging from the ceiling. The total length of the hallway consists of 10 discrete locations where the robot can be located. The robot can only move along the hallway in discrete steps, thus at any point in time it will always be located in one of the 10 possible locations. For simplicity, let's say that the robot can move forward indefinitely. Thus, is just wraps around to location 0 if it drives past location 9.
# 
# The only sensor mounted on the robot is a simple light sensor that point towards the ceiling and it will detect if there is a light above or not. The lights are placed above location 0, 3 and 7 respectively.
# 
# It will look something like this.
# 
# ```{figure} fig/robot_bayes.png
# :name: robot_bayes
# 
# Robot in the hallway
# ```
# 
# It is obvious from the figure that there will be more than one location that will correspond to the measurement from the light sensor, thus the location of the robot is ambigous. Further, the sensor is not perfect so there may also be a small probability of false measurements. That is the sensor may give a reading that there is a light above, when in fact there is no light there. In addition there will may also be a small probability that the odometry makes the robot drive too far or too short as it moves between the locations.
# 
# How do we approach this problem?
# 
# This is were Bayes Filter comes into play. Let's start by defining the initial probability that the robot is located in one of the 10 possible locations. In robotics this probability is called a *belief*. When we start the filter, it is equally likely that the robot is in any of the possible locations. And since the sum of all the probabilities must sum up to unity, the inital belief that the robot is located in one of the locations is 0.1.

# In[1]:


from numpy import array

# Initial belief
bel = array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
print(f"Sum of probabilities: {sum(bel):.1f}")


# The robots knowledge of its surroundings is called a *map*. In our example the robots map will be the location of the lights in the ceiling. As the robots drives through the hallway, the sensor will detect the lights and the belief is updated accordingly.
# 
# So, given that the sensor has detected a light the robot is equally likely to be in location 0, 3 or 7. Or put in other words, it is a probability of 33% that the robot is in location 0, 3, or 7.

# In[2]:


# World map
world_map = array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0])

# Updated belief
bel = array([0.333, 0, 0, 0.333, 0, 0, 0, 0.333, 0, 0])
print(f"Sum of probabilities: {sum(bel):.1f}")


# This belief would be the output of the Bayes Filter if the sensor readings and control commands are both assumed to be perfect.

# (Bayes:correct)=
# ## Correction
# Let us now use the generic correction step from the Bayes Filter algorithm to compute the belief. It may look something like this.

# In[3]:


from numpy import ones

# Initial belief
bel = 1/10*ones(10)

# Sensor reading
z = array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0])

# Normalize
def normalize(x):
    return x/sum(x)

# Correction step
def correct_step(z, bel):
    return normalize(z*bel)

# Main
print("Initial:   " + ' '.join(f'{bel:.3f}' for bel in bel))

belc = correct_step(z[0] == world_map, bel)
print("Corrected: " + ' '.join(f'{belc:.3f}' for belc in belc))


# (Bayes:predict)=
# ## Prediction
# The prediction step consists of moving the robot one location to the right, thus shifting the current belief with the same amount. If we implement the prediction step and run the filter, we obtain the following beliefs.

# In[4]:


from numpy import roll

# Prediction step
def predict_step(bel):
    return roll(bel, 1)

# Main loop
print("Initial: " + ' '.join(f'{bel:.3f}' for bel in bel))

for n in range(0, 8):
    print("Loc: ", n)
    
    bel = predict_step(bel)
    print("Predict: " + ' '.join(f'{bel:.3f}' for bel in bel))
    
    bel = correct_step(z[n] == world_map, bel)
    print("Correct: " + ' '.join(f'{bel:.3f}' for bel in bel))


# Thus, after the correcting the belief in location 6 the robot concludes with a probability of 100% that it must be in location 6 (i.e. the probability is 1 for location 6). This makes perfectly sense as there now are no longer any ambiguities concerning the true location of the robot since the current sensor reading is "no light", i.e. at this step location 6 is the only possible location that fits the sensor readings. From there on the robot is absolute certain of its location.
# 
# This example illustrates the behaviour of an ideal Bayes Filter, i.e. no measurement errors and no prediction errors.

# ## Sensor Noise and Movement Noise
# Unfortunately there is no such thing as a perfect sensor - they are all prone to errors. Therefore we need to account for *measurement errors* due to these sensor imperfections. One way to do so is to introduce the probability that the sensor reading is correct. This lead us to the concept of *likelihood*. We can model this as a vector that represents the likelihood that the sensor readings are correct. Thus, we provide a arbitrary low value for the locations were the probability of seeing a light is low, and a arbitrary high value where the probability of seeing a light is high.
# 
# In addition there will also be *prediction errors* due to imperfections in the motor controls that makes the robot move to far or too short. The mathematical operation that combines two probability density functions into one is called *convolution*. As a result, we obtain a new probability density function that express how the shape of one distribution "blends" with the other distribution. In our case, we are combining the probability distribution function of a location belief with the probability density function (kernel) that models a noisy movement.
# 
# We can now modify our algorithm as follows.

# In[5]:


from numpy import convolve

# Initial belief
bel = 1/10*ones(10)

# Probability of correct sensor reading
z_prob = 0.9

# Noisy movement (kernel)
kernel = array([0.1, 0.8, 0.1])

def likelihood(world_map, z, z_prob):
    likelihood = ones(len(world_map))
    for ind, val in enumerate(world_map):
        if val == z:
            likelihood[ind] = z_prob
        else:
            likelihood[ind] = (1 - z_prob)

    return likelihood

# Prediction step
def predict_step(bel):
    bel = roll(bel, 1)
    return convolve(kernel, bel, 'same')


# Main loop
#set_printoptions(precision=3, suppress = True)
print("Initial: ", bel)

for n in range(0, 8):
    print("Loc: ", n)
    
    bel = predict_step(bel)
    print("Predict: " + ' '.join(f'{bel:.3f}' for bel in bel))
    
    bel = correct_step(likelihood(world_map, z[n], z_prob), bel)
    print("Correct: " + ' '.join(f'{bel:.3f}' for bel in bel))


# So, after the correcting the belief in location 7 the robot concludes with a probability of about 66% that it must be in location 7 (i.e. the probability is 0.66 for location 7). So, even after the introduction of both sensor noise and movement noise, the robot is still able to conclude with a significant probability that it is in the correct location.
# 
# Note that this probability will increase even further if the robot are allowed to drive through the hallway multiple times.

# ## Summary
# The Bayes Filter assumes the current state is a complete summary of the past, thus this assumption implies that the belief is sufficient to represent the history - this is called a *complete state*.
# 
# ```{note}
# The *Markov Assumption* postulates that past and future data are independent if one knows the current state.
# ```
# As stated earlier, the Bayes Filter is not a practical algorithm as it cannot be implemented on a computer. One of the most famous realizations of this filter is the *Kalman Filter*.
# 
# So, now lets's move on to the Kalman Filter - the main workhorse of sensor fusion...
