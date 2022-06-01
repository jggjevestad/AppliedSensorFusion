#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ### Example 5.3
# But what happens if we do not have a laser scanner, but the only sensor is a speedometer? Can we still get estimates of both position and velocity? Let's assume the same dynamic model as in the previous example, but this time with only a speedometer available.
# 
# ```{figure} fig/speed.png
# :name: speedometer
# 
# ```
# 
# The dynamic model will is the same as before, but we write it again here for completness.
# 
# $$
# \begin{bmatrix}
#   \dot{x}\\
#   \ddot{x}\\
# \end{bmatrix}
# =
# \begin{bmatrix}
#   0 &1\\
#   0 &0\\
# \end{bmatrix}
# \begin{bmatrix}
#   x\\
#   \dot{x}\\
# \end{bmatrix}
# +
# \begin{bmatrix}
#   0\\
#   \sqrt{q_v}\\
# \end{bmatrix}
# u
# $$
# 
# Since the only measurements involved are the velocity measurements from the speedometer, the measurement model can be written like this.
# 
# $$
# \begin{bmatrix}
#   z
# \end{bmatrix}
# =
# \begin{bmatrix}
#   0 &1\\
# \end{bmatrix}
# \begin{bmatrix}
#   x\\
#   \dot{x}\\
# \end{bmatrix}
# +
# \begin{bmatrix}
#   v
# \end{bmatrix}
# $$
