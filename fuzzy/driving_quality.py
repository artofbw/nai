"""
It is program that compute quality of driving
Authors:
Maciej Rybacki
Łukasz Ćwikliński

* Antecednets (Inputs)
   - `speedRatio`
      * Universe: How good was the speed ratio, on scale of 0 to 10?
      * Fuzzy set: poor, average, good
   - `fuelUsageRatio`
      * Universe: How good was the speed fuel usage ratio, on scale of 0 to 10?
      * Fuzzy set: poor, average, good
   - `drivingDynamicsRatio`
      * Universe: How good was the driving dynamics ratio, on scale of 0 to 10?
      * Fuzzy set: poor, average, good
* Consequents (Outputs)
   - `quality`
      * Universe: How much quality you get
      * Fuzzy set: poor, average, good
* Rules
rule1 = ctrl.Rule(speedRatio['poor'] | fuelUsageRatio['poor'] | drivingDynamicsRatio['poor'], quality['poor'])
rule2 = ctrl.Rule(speedRatio['good'] | fuelUsageRatio['average'] & drivingDynamicsRatio['average'], quality['average'])
rule3 = ctrl.Rule(speedRatio['average'] & fuelUsageRatio['average'] | drivingDynamicsRatio['good'], quality['average'])
rule4 = ctrl.Rule(speedRatio['average'] & drivingDynamicsRatio['average'] | fuelUsageRatio['good'], quality['average'])
rule5 = ctrl.Rule(speedRatio['average'] & fuelUsageRatio['average'] & drivingDynamicsRatio['good'], quality['good'])
rule6 = ctrl.Rule(speedRatio['good'] & fuelUsageRatio['average'] & drivingDynamicsRatio['good'], quality['good'])
rule7 = ctrl.Rule(speedRatio['average'] & fuelUsageRatio['good'] & drivingDynamicsRatio['good'], quality['good'])
rule8 = ctrl.Rule(speedRatio['good'] & fuelUsageRatio['good'] & drivingDynamicsRatio['good'], quality['good'])
"""
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# New Antecedent/Consequent objects hold universe variables and membership
speedRatio = ctrl.Antecedent(np.arange(0, 11, 1), 'speedRatio')
fuelUsageRatio = ctrl.Antecedent(np.arange(0, 11, 1), 'fuelUsageRatio')
drivingDynamicsRatio = ctrl.Antecedent(np.arange(0, 11, 1), 'drivingDynamicsRatio')
quality = ctrl.Consequent(np.arange(0, 11, 1), 'quality')

# Auto-membership function population is possible with .automf(3)
speedRatio.automf(3)
fuelUsageRatio.automf(3)
drivingDynamicsRatio.automf(3)

# Custom membership functions can be built interactively with a familiar
quality['poor'] = fuzz.trimf(quality.universe, [0, 0, 6])
quality['average'] = fuzz.trimf(quality.universe, [0, 6, 10])
quality['good'] = fuzz.trimf(quality.universe, [6, 10, 10])

# You can see how these look with .view()
speedRatio.view()
fuelUsageRatio.view()
drivingDynamicsRatio.view()

quality.view()

rule1 = ctrl.Rule(speedRatio['poor'] | fuelUsageRatio['poor'] | drivingDynamicsRatio['poor'], quality['poor'])
rule2 = ctrl.Rule(speedRatio['good'] | fuelUsageRatio['average'] & drivingDynamicsRatio['average'], quality['average'])
rule3 = ctrl.Rule(speedRatio['average'] & fuelUsageRatio['average'] | drivingDynamicsRatio['good'], quality['average'])
rule4 = ctrl.Rule(speedRatio['average'] & drivingDynamicsRatio['average'] | fuelUsageRatio['good'], quality['average'])
rule5 = ctrl.Rule(speedRatio['average'] & fuelUsageRatio['average'] & drivingDynamicsRatio['good'], quality['good'])
rule6 = ctrl.Rule(speedRatio['good'] & fuelUsageRatio['average'] & drivingDynamicsRatio['good'], quality['good'])
rule7 = ctrl.Rule(speedRatio['average'] & fuelUsageRatio['good'] & drivingDynamicsRatio['good'], quality['good'])
rule8 = ctrl.Rule(speedRatio['good'] & fuelUsageRatio['good'] & drivingDynamicsRatio['good'], quality['good'])

"""
Control System Creation with provided rules
"""

qualityCheck_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])

"""
Create a ControlSystemSimulation. 
"""

qualityCheck = ctrl.ControlSystemSimulation(qualityCheck_ctrl)

"""
Providing input values
"""
qualityCheck.input['speedRatio'] = 10
qualityCheck.input['fuelUsageRatio'] = 10
qualityCheck.input['drivingDynamicsRatio'] = 10

qualityCheck.compute()

print(qualityCheck.output['quality'])
quality.view(sim=qualityCheck)


plt.show()
