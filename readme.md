# LPV-MPC Vehicle Control Project

This project implements a **Linear Parameter-Varying (LPV) Model Predictive Controller (MPC)** for a nonlinear vehicle model.  
The goal is to track reference trajectories such as **straight‑line motion** and **circular paths** while respecting the vehicle dynamics.

The code simulates:
- Vehicle state evolution using a dynamic model  
- LPV linearization at each time step  
- MPC optimization over a prediction horizon  
- Tracking of reference trajectories  
- Comparison of actual vs. desired motion  

The formulation is based on this course however the implementation is different: # LPV-MPC Vehicle Control Project

This project implements a **Linear Parameter-Varying (LPV) Model Predictive Controller (MPC)** for a nonlinear vehicle model.  
The goal is to track reference trajectories such as **straight‑line motion** and **circular paths** while respecting the vehicle dynamics.

The code simulates:
- Vehicle state evolution using a dynamic model  
- LPV linearization at each time step  
- MPC optimization over a prediction horizon  
- Tracking of reference trajectories  
- Comparison of actual vs. desired motion  
---

## 🎯 Goal

The main goal is to show how LPV-MPC can control a vehicle following different trajectories while handling nonlinearities of the system.  
The project includes:

- Straight‑line path tracking  
- Circle trajectory tracking  
- Trajectory generation utilities  
- Performance metrics and plotting

This demonstrates how MPC can stabilize and guide a vehicle under varying conditions.

---

## 🔧 Method (Short Explanation)

1. The **nonlinear vehicle** is represented by a 6‑state model.  
2. At each simulation step:
   - The model is linearized around the current state → creating an **LPV** model.
   - MPC uses this linear model to predict future behavior over a short horizon.
   - The controller solves an optimization problem to compute:
     - steering angle  
     - longitudinal force  
3. The system evolves using the chosen input, and the loop repeats.

here you can see the results for circular path. Future implementation tries to impose constraints to the controller. 

![Simulation Results](results1.jpg)

---

## 📚 Acknowledgment

The formulation of this project is inspired by the following course:

- [Applied Control Systems 2: Autonomous Cars (360° Tracking)](https://www.udemy.com/course/applied-control-systems-2-autonomous-cars-360-tracking/?couponCode=MT260427G1)

While the overall control framework follows similar principles, the implementation presented in this repository is independently developed and differs from the original course material.