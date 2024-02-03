import numpy as np
from geometry import * 
from threejs_group import *

if __name__ == "__main__":

  red="0xff0000"
  green="0x00ff00"
  purple="0xff00ff"
  blue="0x0000ff"

  trajectory0 = []
  trajectory1 = []
  trajectory_obs = []
  viz_out = threejs_group(js_dir="../js")

  # state: [t, position, quaternion, color] 
  for t in np.arange(0,10,0.1,dtype=float):
    color = red if t < 5 else green
    state0 =[t, [t,0,0.5], [1,0,0,0], color]
    state1 =[t, [t,1.5*t,1], [1,0,0,0], purple]
    trajectory0.append(state0)
    trajectory1.append(state1)

  box0 = box("box0", 1,1,1, trajectory0[0][1], trajectory0[0][2])
  geom = sphere("sphere_0", 1, trajectory1[0][1], trajectory1[0][2])

  viz_out.add_animation(box0, trajectory0)
  viz_out.add_animation(geom, trajectory1)
  viz_out.to_html("animation.html", "out/");
