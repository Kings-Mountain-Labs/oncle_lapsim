---
marp: true
theme: uncover
class:
  - lead
  - invert
math: mathjax
---

# Fitting a MFTire empirical tire model

By Ian

---

# Introduction

---

Variable Definitions

$\alpha$ - slip angle
$\kappa$ - slip ratio
$F_{x}$ - longitudinal force
$F_{y}$ - lateral force
$F_{z}$ - vertical force
$\gamma$ - inclination angle
$p_{i}$ - pressure
$V_{x}$ - forward velocity

---
The basic Magic Formula with pure slip
<br>
$$
F_{xp} = D_{x}sin(C_{x}arctan(B_{x}\kappa_{x}-E_{x}(B_{x}\kappa_{x}-arctan(B_{x}\kappa_{x}))))+S_{Vx}
$$

$$
F_{yp} = D_{y}sin(C_{y}arctan(B_{y}\alpha_{y}-E_{y}(B_{y}\alpha_{y}-arctan(B_{y}\alpha_{y}))))+S_{Vy}
$$
<br>

where $_{p}$ denotes pure slip

---

## Combined slip
<br>

$$
F_{x} = (F_{xp}\cdot G_{x\alpha})
$$
$$
F_{y} = (F_{yp}\cdot G_{y\kappa})+S_{Vy\kappa}
$$
<br>

where $G_{x\alpha}$ and $G_{y\kappa}$ are the combined slip factors
and $S_{Vy\kappa}$ is the combined slip shift for the lateral force

---

# Actually fitting the model

---

## Some sudo code
```python
tire_model # The tire model object
data # a pandas dataframe with the data
# for this example it is all the TTC runs for the particular tire joined together
param_mask # a list of the indexes of the coefficients to be fit in this step

def fx_loss(params, data, tire_model, mask):
    tire_model.params[mask] = params
    new_data = tire_model.predict(data)
    return np.abs(data.FX - new_data.FX)

least_squares(fx_loss, tire_model.dump_params()[mask], args=(data, tire_model, mask,), method='trf', jac='2-point', verbose=2, ftol=0.001)
```
<br>

Process:
1. Select a subset of the coefficients to fit
2. Find appropriate subset of data to fit
3. Run least squares to find the best fit
4. Repeat for all subsets of coefficients

---
## So how do we select which coefficients to fit and which data to go with it?

---

Expanding out the equations
<br>

$$
D_{x}=\mu_{x}F_{z}
$$
$$
\mu_{x} = (P_{Dx1} + P_{Dx2}df_{z})(1-P_{Dx3}\gamma^{2})(1 + P_{px3}dp_{i}^{2})  \lambda_{\mu s}
$$

