import Sofa.Core
import Sofa.constants.Key as Key
import numpy as np

class cablekeyboardcontroller(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, args, kwargs)
        self.cable1 = args[0]
        self.cable2 = args[1]
        self.cable3 = args[2]
        self.cable4 = args[3]
        self.name = "cablecontrol"

    def onKeypressedEvent(self, e):
        displacement1 = self.cable1.CableConstraint.value[0]
        displacement2 = self.cable2.CableConstraint.value[0]
        displacement3 = self.cable3.CableConstraint.value[0]
        displacement4 = self.cable4.CableConstraint.value[0]
        print(f"displacement1 is {displacement1}")
        print(f"displacement2 is {displacement2}")
        print(f"displacement3 is {displacement3}")
        print(f"displacement4 is {displacement4}")

        print(self.cable1.getRoot().ElasticMaterialObject2.dofs.position.value[6])
        print(self.cable1.getRoot().ElasticMaterialObject2.dofs.position.value[0])
        print(self.cable1.getRoot().ElasticMaterialObject2.dofs.position.value[452])
        print(self.cable1.getRoot().ElasticMaterialObject2.dofs.position.value[9])
        c1 = np.array(self.cable1.getRoot().ElasticMaterialObject2.dofs.position.value[6])
        c2 = np.array(self.cable1.getRoot().ElasticMaterialObject2.dofs.position.value[0])
        c3 = np.array(self.cable1.getRoot().ElasticMaterialObject2.dofs.position.value[452])
        c4 = np.array(self.cable1.getRoot().ElasticMaterialObject2.dofs.position.value[9])
        center = (c1 + c2 + c3 + c4) / 4
        y_new = c4 - c3
        y_new = y_new / np.linalg.norm(y_new)
        z_new = c1 - c2
        z_new = z_new / np.linalg.norm(z_new)
        x_new = np.cross(y_new, z_new)
        x_new = x_new / np.linalg.norm(x_new)
        print("Square Center (New Frame Origin):", center)
        print("New X-axis:", x_new)
        print("New Y-axis:", y_new)
        print("New Z-axis:", z_new)
        target = np.array([41.3192, -2.83512, 2.2299])
        translated_target = target - center
        target_new = np.array([np.dot(translated_target, x_new),
                            np.dot(translated_target, y_new),
                            np.dot(translated_target, z_new)])
        print("Target in New Frame Coordinates:", target_new)

        # bending upwards
        if e["key"] == Key.KP_1:
            if displacement3 > 0: # to reduce unstable issue
                displacement3 -= 0.1
            else:
                if self.cable4.CableConstraint.cableLength.value > 13:
                    displacement4 += 0.1

        # bending downwards
        elif e["key"] == Key.KP_2:
            if displacement4 > 0: # to reduce unstable issue
                displacement4 -= 0.1
            else:
                if self.cable3.CableConstraint.cableLength.value > 13:
                    displacement3 += 0.1

        # bending leftwards
        elif e["key"] == Key.KP_3:
            if displacement2 > 0: # to reduce unstable issue
                displacement2 -= 0.1
            else:
                if self.cable1.CableConstraint.cableLength.value > 13:
                    displacement1 += 0.1

        # bending rightwards
        elif e["key"] == Key.KP_4:
            if displacement1 > 0: # to reduce unstable issue
                displacement1 -= 0.1
            else:
                if self.cable2.CableConstraint.cableLength.value > 13:
                    displacement2 += 0.1

        self.cable1.CableConstraint.value = [displacement1]
        self.cable2.CableConstraint.value = [displacement2]
        self.cable3.CableConstraint.value = [displacement3]
        self.cable4.CableConstraint.value = [displacement4]

import numpy as np
class cablestepcontroller(Sofa.Core.Controller):
# continuous version
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, args, kwargs)
        self.cable1 = args[0]
        self.cable2 = args[1]
        self.cable3 = args[2]
        self.cable4 = args[3]
        self.name = "cablestepcontrol"

    def applyAction(self, action):
        """
        Applies an action based on input action like [-0.9552726   0.6214838  -0.36833474]
        """
        if action is not None:
            # .value is a displacement vector stored in that contraint, initialized as 0.
            displacement1 = self.cable1.CableConstraint.value[0]
            displacement2 = self.cable2.CableConstraint.value[0]
            displacement3 = self.cable3.CableConstraint.value[0]
            displacement4 = self.cable4.CableConstraint.value[0]
            
            # bending actuation = [a,b], if a > 0, BU, if a < 0, BD, if 
            # bending upwards
            if action[0] >= 0:
                if displacement3 > 0: # to reduce unstable issue
                    displacement3 -= action[0]
                else:
                    if self.cable4.CableConstraint.cableLength.value > 12.5:
                        displacement4 += action[0]
            if action[0] < 0 :
                # bending downwards
                if displacement4 > 0: # to reduce unstable issue
                    displacement4 += action[0]
                else:
                    if self.cable3.CableConstraint.cableLength.value > 12.5:
                        displacement3 -= action[0]
            if action[1] >=0 :
                # bending leftwards
                if displacement2 > 0: # to reduce unstable issue
                    displacement2 -= action[1]
                else:
                    if self.cable1.CableConstraint.cableLength.value > 12.5:
                        displacement1 += action[1]
            if action[1] < 0 :
            # bending rightwards
                if displacement1 > 0: # to reduce unstable issue
                    displacement1 += action[1]
                else:
                    if self.cable2.CableConstraint.cableLength.value > 12.5:
                        displacement2 -= action[1]
                        
            # print(f"self.cable.CableConstraint.value is {self.cable1.CableConstraint.cableLength.value} ,{self.cable2.CableConstraint.cableLength.value},{self.cable3.CableConstraint.cableLength.value} ,{self.cable4.CableConstraint.cableLength.value}")
            self.cable1.CableConstraint.value = [displacement1]
            self.cable2.CableConstraint.value = [displacement2]
            self.cable3.CableConstraint.value = [displacement3]
            self.cable4.CableConstraint.value = [displacement4]