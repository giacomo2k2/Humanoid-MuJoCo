import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# load xml
xml_path = './models/unitree_g1/g1.xml'


tree = ET.parse(xml_path)
root = tree.getroot()
body = root.find('worldbody/body')
#actuators = root.find('actuator')

#  plot all the positions of the bodies
bodies = {}

for body in root.iter('body'):
    name = body.get('name')
    pos = body.get('pos')
    if pos is not None:
        pos = [float(x) for x in pos.split()]
        bodies[name] = pos
        
# plot the positions of the bodies
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for name, pos in bodies.items():
    ax.scatter(pos[0], pos[1], pos[2], label=name)  

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()


 


