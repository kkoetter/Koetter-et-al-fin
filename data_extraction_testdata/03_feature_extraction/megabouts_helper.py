import matplotlib.pyplot as plt


labels_cat = ['approach_swim',
 'slow1',
 'slow2',
 'burst_swim',
 'J_turn',
 'high_angle_turn',
 'routine_turn',
 'spot_avoidance_turn',
 'O_bend',
 'long_latency_C_start',
 'C_start']

color =  ['#82cfff',
  '#4589ff',
  '#0000c8',
  '#fcaf6d',
  '#ffb3b8',
  '#08bdba',
  '#24a148',
  '#9b82f3',
  '#ee5396',
  '#e3bc13',
  '#fa4d56']


labels_cat_new = ['approach_swim',
  'slow1',
  'slow2',
  'burst_swim',
  'J_turn',
  'high_angle_turn',
  'routine_turn',
  'spot_avoidance_turn',
  'O_bend',
  'long_latency_C_start',
  'short_latency_C_start']

#Load the Set2 colormap
cmap = plt.get_cmap('Set2')

#Get two colors from the colormap
color1 = cmap(0) # Get the first color
color2 = cmap(3) # Get the second color
color3 = cmap(7) # Get the second color

print("Color 1:", color1)
print("Color 2:", color2)

color_ipsi_cont = [color1, color2, color3]